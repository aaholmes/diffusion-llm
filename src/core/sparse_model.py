#!/usr/bin/env python3
"""
Sparse Denoiser Models for SDD (Sparse Distribution Diffusion).

This module implements two denoiser architectures:

1. SparseDenoiser (v1):
   - Aggregates k candidates to single embedding per position
   - Standard transformer processing
   - Simpler, faster, baseline

2. BilateralSparseDenoiser (v2):
   - Preserves [B, L, k, D] tensor throughout
   - Intra-position attention: candidates interact within each position
   - Inter-position attention: positions interact via pooled representations
   - More expressive, better quality

Architecture Overview:
    Input: SparseState [B, L, k, E]
        ↓
    Probability encoding + position/time embeddings
        ↓
    (v1) Aggregate to [B, L, D] → Standard transformer
    (v2) Bilateral attention blocks on [B, L, k, D]
        ↓
    Output projection: [B, L, vocab_size]
        ↓
    Top-k → New SparseState (for chaining)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.sparse_diffusion import SparseState, SDDConfig


@dataclass
class SparseModelConfig:
    """Configuration for Sparse Denoiser model."""
    # Vocabulary and embeddings
    vocab_size: int = 8192
    embed_dim: int = 64      # Embedding dimension (compact)
    k: int = 8               # Number of top-k tokens per position
    normalize_embeddings: bool = True  # Normalize embeddings to unit sphere

    # Transformer architecture
    d_model: int = 768       # Transformer hidden dimension
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048         # Feed-forward dimension
    dropout: float = 0.1
    max_seq_len: int = 128

    # Conditioning (for image captioning, None for unconditional)
    encoder_dim: Optional[int] = 768  # Dimension of encoder output (e.g., CLIP)
    encoder_seq_len: int = 50  # Length of encoder output

    # Special tokens
    pad_token_id: int = 0
    ignore_token_id: int = 5


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional/timestep embeddings."""

    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim

        # Precompute positional encodings
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Position indices [batch, seq_len] or timesteps [batch]

        Returns:
            Embeddings [batch, seq_len, dim] or [batch, dim]
        """
        if x.dim() == 1:
            # Timesteps: scale to [0, 1000] range for good spread
            indices = (x * 1000).long().clamp(0, self.pe.shape[0] - 1)
            return self.pe[indices]  # [batch, dim]
        else:
            # Positions
            return self.pe[:x.shape[1]]  # [seq_len, dim]


class TransformerBlock(nn.Module):
    """
    Transformer block with optional cross-attention.

    Architecture (pre-norm):
        x → LayerNorm → SelfAttention → + → LayerNorm → CrossAttention → + → LayerNorm → FFN → +
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        has_cross_attention: bool = True,
    ):
        super().__init__()

        self.has_cross_attention = has_cross_attention

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (optional)
        if has_cross_attention:
            self.norm2 = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, d_model]
            encoder_output: Encoder output for cross-attention [batch, enc_len, d_model]
            encoder_mask: Encoder padding mask [batch, enc_len]

        Returns:
            Output [batch, seq_len, d_model]
        """
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout1(attn_out)

        # Cross-attention (if encoder output provided)
        if self.has_cross_attention and encoder_output is not None:
            normed = self.norm2(x)
            # Convert mask: 1 = valid, 0 = ignore → key_padding_mask: True = ignore
            key_padding_mask = None
            if encoder_mask is not None:
                key_padding_mask = encoder_mask == 0
            cross_out, _ = self.cross_attn(
                normed, encoder_output, encoder_output,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout2(cross_out)

        # Feed-forward
        normed = self.norm3(x)
        x = x + self.ffn(normed)

        return x


# =============================================================================
# SDD v2: Full Sparse Attention Architecture
# =============================================================================

class FullSparseAttention(nn.Module):
    """
    Full attention across ALL candidates at ALL positions.

    Flattens [B, L, k, D] → [B, L*k, D], does self-attention,
    then reshapes back. Uses probability biasing and sinusoidal position embeddings.

    This allows:
    - Candidate 0 at position 5 to attend to candidate 3 at position 12
    - Natural competition between candidates across positions
    - Coherent token selection via full context

    Input: [B, L, k, D]
    Output: [B, L, k, D]
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k = k
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Probability bias scale
        self.prob_bias_scale = nn.Parameter(torch.tensor(1.0))

        # Use sinusoidal position embeddings for flexibility with varying L*k
        # This handles dynamic sequence lengths and k values
        self.pos_emb = SinusoidalEmbedding(d_model)

    def forward(self, h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, L, k, D]
            probs: [B, L, k]
        Returns:
            [B, L, k, D]
        """
        B, L, k, D = h.shape

        # Flatten to [B, L*k, D]
        h_flat = h.view(B, L * k, D)
        probs_flat = probs.view(B, L * k)  # [B, L*k]

        # Add sinusoidal position embeddings (handles any L*k)
        # SinusoidalEmbedding forward with 2D input returns [seq_len, D]
        # Create a dummy batch dimension to get the right shape
        positions = torch.zeros(1, L * k, device=h.device, dtype=torch.long)
        pos_emb = self.pos_emb(positions)  # [1, L*k, D] -> we just need [L*k, D]
        h_flat = h_flat + pos_emb

        # Project Q, K, V
        q = self.q_proj(h_flat)  # [B, L*k, D]
        key = self.k_proj(h_flat)
        v = self.v_proj(h_flat)

        # Reshape for multi-head attention
        q = q.view(B, L * k, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, L*k, head_dim]
        key = key.view(B, L * k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L * k, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, heads, L*k, L*k]
        scores = torch.matmul(q, key.transpose(-1, -2)) * self.scale

        # Add probability bias (attend more to high-prob candidates)
        prob_bias = torch.log(probs_flat + 1e-8)  # [B, L*k]
        prob_bias = prob_bias.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L*k]
        scores = scores + self.prob_bias_scale * prob_bias

        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B, heads, L*k, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L * k, D)
        out = self.out_proj(out)

        # Reshape back to [B, L, k, D]
        return out.view(B, L, k, D)


# Legacy classes kept for backwards compatibility with existing tests
class IntraPositionAttention(nn.Module):
    """
    Attention within each position's k candidates.

    Each position is independent - candidates only see other candidates
    at the same position. Probability bias encourages attending to
    high-probability candidates.

    NOTE: This is the legacy implementation. New code should use FullSparseAttention.

    Input: [B, L, k, D]
    Output: [B, L, k, D]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable scale for probability bias
        self.prob_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, L, k, D]
            probs: [B, L, k]

        Returns:
            [B, L, k, D]
        """
        B, L, k, D = h.shape

        # Reshape to treat each position independently: [B*L, k, D]
        h_flat = h.view(B * L, k, D)
        probs_flat = probs.view(B * L, k)

        # Project Q, K, V
        q = self.q_proj(h_flat)  # [B*L, k, D]
        key = self.k_proj(h_flat)
        v = self.v_proj(h_flat)

        # Reshape for multi-head attention: [B*L, n_heads, k, head_dim]
        q = q.view(B * L, k, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(B * L, k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * L, k, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B*L, n_heads, k, k]
        scores = torch.matmul(q, key.transpose(-1, -2)) * self.scale

        # Add probability bias (key-only): attend more to high-prob candidates
        # log(prob) before softmax ≡ multiplying attention by prob after softmax
        prob_bias = torch.log(probs_flat + 1e-8)  # [B*L, k]
        prob_bias = prob_bias.unsqueeze(1).unsqueeze(2)  # [B*L, 1, 1, k]
        scores = scores + self.prob_bias_scale * prob_bias

        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B*L, n_heads, k, head_dim]
        out = out.transpose(1, 2).contiguous().view(B * L, k, D)
        out = self.out_proj(out)

        # Reshape back: [B, L, k, D]
        return out.view(B, L, k, D)


class InterPositionAttention(nn.Module):
    """
    Attention across positions (L×L), using pooled representations.

    Pools each position's k candidates (prob-weighted), does standard
    self-attention across positions, then broadcasts back to all candidates.

    NOTE: This is the legacy implementation. New code should use FullSparseAttention.

    Input: [B, L, k, D]
    Output: [B, L, k, D]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, L, k, D]
            probs: [B, L, k]

        Returns:
            [B, L, k, D]
        """
        B, L, k, D = h.shape

        # Pool candidates at each position using probability weights
        weights = probs.unsqueeze(-1)  # [B, L, k, 1]
        h_pooled = (h * weights).sum(dim=2)  # [B, L, D]

        # Self-attention across positions
        h_attn, _ = self.attn(h_pooled, h_pooled, h_pooled)  # [B, L, D]

        # Broadcast back to all candidates
        h_attn_broadcast = h_attn.unsqueeze(2).expand(-1, -1, k, -1)  # [B, L, k, D]

        return h_attn_broadcast


class BilateralSparseBlock(nn.Module):
    """
    Transformer block with full sparse attention.

    Architecture (pre-norm):
        h → LayerNorm → FullSparseAttn → +
        h → LayerNorm → CrossAttn(encoder) → +  (if encoder provided)
        h → LayerNorm → FFN → +

    Uses full (L×k) attention where every candidate at every position can
    attend to all other candidates, enabling coherent token selection.

    Maintains [B, L, k, D] throughout.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        k: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Full attention across all L*k candidates
        self.norm1 = nn.LayerNorm(d_model)
        self.full_attn = FullSparseAttention(d_model, n_heads, max_seq_len, k, dropout)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to encoder
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # FFN (applied to each candidate)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        probs: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, L, k, D]
            probs: [B, L, k]
            encoder_output: [B, enc_len, D]
            encoder_mask: [B, enc_len]

        Returns:
            h: [B, L, k, D]
        """
        B, L, k, D = h.shape

        # 1. Full attention (replaces both intra and inter)
        h = h + self.dropout1(self.full_attn(self.norm1(h), probs))

        # 2. Cross-attention if encoder provided
        if encoder_output is not None:
            # Flatten for cross-attention: [B, L*k, D]
            h_flat = h.view(B, L * k, D)
            h_normed = self.norm2(h_flat)

            # Cross-attention
            key_padding_mask = None
            if encoder_mask is not None:
                key_padding_mask = encoder_mask == 0

            h_cross, _ = self.cross_attn(
                h_normed, encoder_output, encoder_output,
                key_padding_mask=key_padding_mask,
            )
            # Reshape back and add residual
            h = h + self.dropout2(h_cross.view(B, L, k, D))

        # 3. FFN (applied to each candidate independently)
        h = h + self.ffn(self.norm3(h))

        return h


class BilateralSparseDenoiser(nn.Module):
    """
    SDD v2: Full Sparse Attention Denoiser.

    Unlike SparseDenoiser which immediately aggregates k candidates into one
    embedding, this model preserves the full [B, L, k, D] tensor throughout
    all transformer layers, allowing candidates to interact via attention.

    Key innovation: Full (L×k) attention
    - Every candidate at every position can attend to all other candidates
    - Enables coherent token selection across positions
    - Natural competition between candidates

    Architecture:
        Input: SparseState [B, L, k, E]
            ↓
        Probability encoding + position/time embeddings
            ↓
        N × BilateralSparseBlock (full_attn + cross + FFN)
            ↓
        Learned readout (collapse k → 1)
            ↓
        Output logits [B, L, vocab_size]
    """

    def __init__(self, config: SparseModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Probability encoder: scalar prob → embedding
        self.prob_encoder = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # Project from embed_dim to d_model
        self.input_proj = nn.Linear(config.embed_dim, config.d_model)

        # Positional embeddings
        self.pos_embedding = SinusoidalEmbedding(config.d_model, config.max_seq_len)

        # Timestep embeddings
        self.time_embedding = SinusoidalEmbedding(config.d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # Encoder projection (for cross-attention, optional)
        if config.encoder_dim is not None:
            self.encoder_proj = nn.Linear(config.encoder_dim, config.d_model)
        else:
            self.encoder_proj = None

        # Full sparse transformer layers
        self.layers = nn.ModuleList([
            BilateralSparseBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.k,
                config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Learned readout query (collapses k → 1)
        self.readout_query = nn.Parameter(torch.randn(1, 1, 1, config.d_model) * 0.02)

        # Learnable scale for readout probability bias
        self.readout_prob_scale = nn.Parameter(torch.tensor(1.0))

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        state: SparseState,
        t: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with bilateral sparse attention.

        Args:
            state: SparseState with probs [B, L, k], indices [B, L, k]
            t: Timesteps [B]
            encoder_output: [B, enc_len, enc_dim]
            encoder_mask: [B, enc_len]

        Returns:
            logits: [B, L, vocab_size]
        """
        B, L, k = state.probs.shape
        probs = state.probs  # Keep for attention biasing

        # 1. Look up token embeddings from indices
        token_emb = self.token_embedding(state.indices)  # [B, L, k, E]

        # Normalize embeddings to unit sphere if configured
        # This makes magnitude encode confidence: certain=on sphere, uncertain=toward origin
        if self.config.normalize_embeddings:
            token_emb = F.normalize(token_emb, p=2, dim=-1)

        # 2. Encode probabilities
        prob_emb = self.prob_encoder(state.probs.unsqueeze(-1))  # [B, L, k, E]

        # 3. Combine token and probability embeddings
        h = token_emb + prob_emb  # [B, L, k, E]

        # 3. Project to model dimension
        h = self.input_proj(h)  # [B, L, k, D]

        # 4. Add positional embeddings (broadcast to k)
        positions = torch.arange(L, device=h.device)
        pos_emb = self.pos_embedding(positions)  # [L, D]
        h = h + pos_emb.view(1, L, 1, -1)  # Broadcast to [B, L, k, D]

        # 5. Add timestep embeddings (broadcast to all)
        time_emb = self.time_embedding(t)  # [B, D]
        time_emb = self.time_proj(time_emb)  # [B, D]
        h = h + time_emb.view(B, 1, 1, -1)  # Broadcast to [B, L, k, D]

        # 6. Project encoder output if provided
        enc_proj = None
        if encoder_output is not None and self.encoder_proj is not None:
            enc_proj = self.encoder_proj(encoder_output)  # [B, enc_len, D]

        # 7. Bilateral sparse transformer layers
        for layer in self.layers:
            h = layer(h, probs, enc_proj, encoder_mask)

        # 8. Final norm
        h = self.final_norm(h)  # [B, L, k, D]

        # 9. Learned readout to collapse k dimension
        # Query attends over k candidates at each position
        readout_q = self.readout_query.expand(B, L, 1, -1)  # [B, L, 1, D]
        scores = torch.matmul(readout_q, h.transpose(-1, -2))  # [B, L, 1, k]
        scores = scores / math.sqrt(self.config.d_model)

        # Add probability bias (key-only)
        prob_bias = torch.log(probs + 1e-8).unsqueeze(2)  # [B, L, 1, k]
        scores = scores + self.readout_prob_scale * prob_bias

        attn_weights = F.softmax(scores, dim=-1)  # [B, L, 1, k]
        h_out = torch.matmul(attn_weights, h).squeeze(2)  # [B, L, D]

        # 10. Output projection
        logits = self.output_proj(h_out)  # [B, L, vocab_size]

        return logits

    def denoise_step(
        self,
        state: SparseState,
        t: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> SparseState:
        """
        Single denoising step: input sparse → output sparse.

        Args:
            state: Input SparseState
            t: Timesteps [batch]
            encoder_output: Encoder output for conditioning
            encoder_mask: Encoder mask
            temperature: Sampling temperature (lower = more confident)

        Returns:
            New SparseState (top-k from output distribution)
        """
        # Get full logits
        logits = self.forward(state, t, encoder_output, encoder_mask)

        # Apply temperature BEFORE softmax (this is the correct way)
        # Lower temperature = sharper distribution (more confident)
        # Higher temperature = flatter distribution (more diverse)
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probs_full = F.softmax(logits, dim=-1)  # [B, L, vocab_size]

        # Get top-k for the sparse state
        k = self.config.k
        top_probs, top_indices = torch.topk(probs_full, k, dim=-1)

        # NOTE: We do NOT renormalize here!
        # The probabilities from softmax already sum to 1 over the full vocab.
        # Taking top-k gives us the k highest probabilities, which correctly
        # represent the model's confidence. Renormalizing would artificially
        # inflate confidence and cause repetition/premature commitment.

        return SparseState(
            probs=top_probs,
            indices=top_indices,
        )

    def get_embedding_table(self) -> nn.Embedding:
        """Return the embedding table for use in diffusion process."""
        return self.token_embedding


class SparseDenoiser(nn.Module):
    """
    SDD v1: Aggregation-based Sparse Distribution Denoiser.

    Takes a sparse distribution state and aggregates k candidates to a single
    embedding per position before processing through standard transformer.

    Simpler and faster than BilateralSparseDenoiser, but less expressive.
    """

    def __init__(self, config: SparseModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (shared with diffusion process)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Project from embed_dim to d_model
        self.input_proj = nn.Linear(config.embed_dim, config.d_model)

        # Positional embeddings
        self.pos_embedding = SinusoidalEmbedding(config.d_model, config.max_seq_len)

        # Timestep embeddings
        self.time_embedding = SinusoidalEmbedding(config.d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # Encoder projection (for cross-attention conditioning, optional)
        if config.encoder_dim is not None:
            self.encoder_proj = nn.Linear(config.encoder_dim, config.d_model)
        else:
            self.encoder_proj = None

        # Transformer layers
        has_cross_attn = config.encoder_dim is not None
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                has_cross_attention=has_cross_attn,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output head: project back to vocabulary logits
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def aggregate_sparse_input(self, state: SparseState) -> torch.Tensor:
        """
        Aggregate sparse distribution to single embedding per position.

        Uses probability-weighted sum (order-invariant).

        Args:
            state: SparseState with probs [B, L, k] and indices [B, L, k]

        Returns:
            Aggregated embeddings [B, L, embed_dim]
        """
        # Look up embeddings from indices
        embeds = self.token_embedding(state.indices)  # [B, L, k, E]

        # Normalize embeddings to unit sphere if configured
        # This makes magnitude encode confidence: certain=on sphere, uncertain=toward origin
        if self.config.normalize_embeddings:
            embeds = F.normalize(embeds, p=2, dim=-1)

        # Weighted sum: sum_j(p_j * e_j)
        # probs: [B, L, k] → [B, L, k, 1]
        weights = state.probs.unsqueeze(-1)
        # embeds: [B, L, k, E]
        # weighted: [B, L, k, E]
        weighted = embeds * weights
        # Sum over k: [B, L, E]
        aggregated = weighted.sum(dim=2)

        return aggregated

    def forward(
        self,
        state: SparseState,
        t: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning full vocabulary logits.

        Args:
            state: Input SparseState
            t: Timesteps [batch]
            encoder_output: Encoder output for conditioning [batch, enc_len, enc_dim]
            encoder_mask: Encoder mask [batch, enc_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size = state.batch_size
        seq_len = state.seq_len

        # 1. Aggregate sparse input to single embedding per position
        aggregated = self.aggregate_sparse_input(state)  # [B, L, embed_dim]

        # 2. Project to model dimension
        h = self.input_proj(aggregated)  # [B, L, d_model]

        # 3. Add positional embeddings
        positions = torch.arange(seq_len, device=h.device)
        pos_emb = self.pos_embedding(positions)  # [L, d_model]
        h = h + pos_emb.unsqueeze(0)

        # 4. Add timestep embeddings
        time_emb = self.time_embedding(t)  # [B, d_model]
        time_emb = self.time_proj(time_emb)  # [B, d_model]
        h = h + time_emb.unsqueeze(1)  # Broadcast to all positions

        # 5. Project encoder output if provided
        enc_proj = None
        if encoder_output is not None and self.encoder_proj is not None:
            enc_proj = self.encoder_proj(encoder_output)  # [B, enc_len, d_model]

        # 6. Transformer layers
        for layer in self.layers:
            h = layer(h, enc_proj, encoder_mask)

        # 7. Final norm and output projection
        h = self.final_norm(h)
        logits = self.output_proj(h)  # [B, L, vocab_size]

        return logits

    def denoise_step(
        self,
        state: SparseState,
        t: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> SparseState:
        """
        Single denoising step: input sparse → output sparse.

        This is what gets chained during generation.

        Args:
            state: Input SparseState
            t: Timesteps [batch]
            encoder_output: Encoder output for conditioning
            encoder_mask: Encoder mask
            temperature: Sampling temperature (lower = more confident)

        Returns:
            New SparseState (top-k from output distribution)
        """
        # Get full logits
        logits = self.forward(state, t, encoder_output, encoder_mask)

        # Apply temperature BEFORE softmax (this is the correct way)
        # Lower temperature = sharper distribution (more confident)
        # Higher temperature = flatter distribution (more diverse)
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probs_full = F.softmax(logits, dim=-1)  # [B, L, vocab_size]

        # Get top-k for the sparse state
        k = self.config.k
        top_probs, top_indices = torch.topk(probs_full, k, dim=-1)  # [B, L, k] each

        # NOTE: We do NOT renormalize here!
        # The probabilities from softmax already sum to 1 over the full vocab.
        # Taking top-k gives us the k highest probabilities, which correctly
        # represent the model's confidence. Renormalizing would artificially
        # inflate confidence and cause repetition/premature commitment.

        return SparseState(
            probs=top_probs,
            indices=top_indices,
        )

    def get_embedding_table(self) -> nn.Embedding:
        """Return the embedding table for use in diffusion process."""
        return self.token_embedding


def create_sparse_model(
    vocab_size: int = 8192,
    embed_dim: int = 64,
    k: int = 8,
    d_model: int = 768,
    n_layers: int = 6,
    n_heads: int = 8,
    encoder_dim: Optional[int] = 768,
    max_seq_len: int = 128,
) -> Tuple[SparseDenoiser, SDDConfig]:
    """
    Create a SparseDenoiser model (v1) with matching config.

    Returns:
        model: SparseDenoiser
        sdd_config: SDDConfig for SparseDiffusion
    """
    model_config = SparseModelConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        k=k,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        encoder_dim=encoder_dim,
        max_seq_len=max_seq_len,
    )

    sdd_config = SDDConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        k=k,
    )

    model = SparseDenoiser(model_config)

    return model, sdd_config


def create_bilateral_sparse_model(
    vocab_size: int = 8192,
    embed_dim: int = 64,
    k: int = 8,
    d_model: int = 768,
    n_layers: int = 6,
    n_heads: int = 8,
    encoder_dim: Optional[int] = 768,
    max_seq_len: int = 128,
) -> Tuple[BilateralSparseDenoiser, SDDConfig]:
    """
    Create a BilateralSparseDenoiser model (v2) with matching config.

    This is the recommended architecture with attention over k candidates.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        k: Number of top-k candidates per position
        d_model: Transformer hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        encoder_dim: Encoder dimension for cross-attention (None for unconditional)
        max_seq_len: Maximum sequence length

    Returns:
        model: BilateralSparseDenoiser
        sdd_config: SDDConfig for SparseDiffusion
    """
    model_config = SparseModelConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        k=k,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        encoder_dim=encoder_dim,
        max_seq_len=max_seq_len,
    )

    sdd_config = SDDConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        k=k,
    )

    model = BilateralSparseDenoiser(model_config)

    return model, sdd_config


# Quick test
if __name__ == "__main__":
    batch_size = 4
    seq_len = 32
    k = 8

    print("=" * 60)
    print("Testing SparseDenoiser (v1 - aggregation)...")
    print("=" * 60)

    # Create model
    model_v1, sdd_config = create_sparse_model(
        vocab_size=8192,
        embed_dim=64,
        k=k,
        d_model=256,
        n_layers=2,
    )

    num_params = sum(p.numel() for p in model_v1.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dummy sparse state
    probs = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)
    indices = torch.randint(0, 8192, (batch_size, seq_len, k))
    state = SparseState(probs=probs, indices=indices)

    # Forward pass
    t = torch.rand(batch_size)
    logits = model_v1(state, t)
    print(f"Logits shape: {logits.shape}")

    # With encoder
    encoder_output = torch.randn(batch_size, 50, 768)
    encoder_mask = torch.ones(batch_size, 50)
    logits_cond = model_v1(state, t, encoder_output, encoder_mask)
    print(f"Conditioned logits shape: {logits_cond.shape}")

    print("v1 tests passed!")

    print("\n" + "=" * 60)
    print("Testing BilateralSparseDenoiser (v2 - attention over k)...")
    print("=" * 60)

    # Create v2 model
    model_v2, sdd_config = create_bilateral_sparse_model(
        vocab_size=8192,
        embed_dim=64,
        k=k,
        d_model=256,
        n_layers=2,
    )

    num_params_v2 = sum(p.numel() for p in model_v2.parameters())
    print(f"Model parameters: {num_params_v2:,}")
    print(f"Parameter increase: {num_params_v2 - num_params:,} ({100*(num_params_v2/num_params - 1):.1f}%)")

    # Create state for v2 (same state, embeddings are looked up internally)
    state_v2 = SparseState(probs=probs, indices=indices)

    # Forward pass
    logits_v2 = model_v2(state_v2, t)
    print(f"Logits shape: {logits_v2.shape}")

    # Denoise step
    new_state = model_v2.denoise_step(state_v2, t)
    print(f"New state probs shape: {new_state.probs.shape}")
    print(f"New state indices shape: {new_state.indices.shape}")

    # With encoder
    logits_v2_cond = model_v2(state_v2, t, encoder_output, encoder_mask)
    print(f"Conditioned logits shape: {logits_v2_cond.shape}")

    # Test gradient flow
    loss = logits_v2_cond.mean()
    loss.backward()
    print(f"Gradients flow correctly: {model_v2.readout_query.grad is not None}")

    print("v2 tests passed!")
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
