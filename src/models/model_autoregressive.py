#!/usr/bin/env python3
"""
Autoregressive transformer for image captioning (baseline for comparison).

Standard encoder-decoder architecture:
- Frozen CLIP image encoder (pre-extracted features)
- Autoregressive decoder with cross-attention
- Left-to-right generation with causal masking
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int, device: str = "cuda") -> torch.Tensor:
        """Generate positional embeddings for sequence length."""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim)
        )

        pe = torch.zeros(seq_len, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to image features
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input [batch, seq_len, d_model]
            encoder_output: Image features [batch, n_patches, d_model]
            self_attn_mask: Causal mask for self-attention [seq_len, seq_len]
            encoder_mask: Mask for encoder [batch, n_patches]
        """
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=self_attn_mask,
            need_weights=False
        )
        x = self.norm1(x + attn_out)

        # Cross-attention to image
        cross_out, _ = self.cross_attn(
            x, encoder_output, encoder_output,
            key_padding_mask=(encoder_mask == 0) if encoder_mask is not None else None,
            need_weights=False
        )
        x = self.norm2(x + cross_out)

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


@dataclass
class AutoregressiveConfig:
    """Configuration for autoregressive captioning model."""
    d_model: int = 768
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    vocab_size: int = 8192
    max_seq_len: int = 128
    dropout: float = 0.1
    pad_token_id: int = 0


class AutoregressiveCaptioner(nn.Module):
    """Autoregressive image captioning model."""

    def __init__(self, config: AutoregressiveConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        # Positional embedding
        self.pos_embed = SinusoidalPositionEmbeddings(config.d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: str) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: Input token IDs [batch, seq_len]
            encoder_output: Image features [batch, n_patches, d_model]
            encoder_mask: Mask for encoder [batch, n_patches] (1=valid, 0=pad)

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Embed tokens
        x = self.token_embed(tokens)  # [batch, seq_len, d_model]

        # Add positional embeddings
        pos_emb = self.pos_embed(seq_len, device)  # [seq_len, d_model]
        x = x + pos_emb.unsqueeze(0)  # Broadcast to [batch, seq_len, d_model]

        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, encoder_mask)

        # Project to vocabulary
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_len: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively.

        Args:
            encoder_output: Image features [batch, n_patches, d_model]
            encoder_mask: Mask for encoder [batch, n_patches]
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token

        Returns:
            generated: Generated token IDs [batch, max_len]
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Start with BOS token
        tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Forward pass
            logits = self.forward(tokens, encoder_output, encoder_mask)

            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        # Pad to max_len if needed
        if tokens.shape[1] < max_len:
            padding = torch.full(
                (batch_size, max_len - tokens.shape[1]),
                self.config.pad_token_id,
                dtype=torch.long,
                device=device
            )
            tokens = torch.cat([tokens, padding], dim=1)

        return tokens
