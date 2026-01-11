#!/usr/bin/env python3
"""
Diffusion Transformer model for discrete diffusion language modeling.

Architecture:
- Bidirectional transformer (no causal masking)
- Timestep conditioning via sinusoidal embeddings
- Pre-norm transformer blocks
- Output projection to vocabulary logits

Usage:
    from model import create_model

    model = create_model("small")  # 25M params
    logits = model(tokens, timestep)  # [batch, seq_len, vocab_size]
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for diffusion timestep.

    Maps scalar timestep t ∈ [0, 1] to a vector of dimension `dim`.
    Uses the same formulation as the original Transformer positional encoding,
    but applied to the continuous timestep value.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep values [batch_size] in range [0, 1]

        Returns:
            Embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        # Frequency bands (log-spaced)
        freq = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        # Outer product of timesteps and frequencies
        # t: [batch_size] -> [batch_size, 1]
        # freq: [half_dim] -> [1, half_dim]
        args = t[:, None] * freq[None, :] * 1000  # Scale for better gradients

        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embeddings


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    Pre-norm applies LayerNorm before attention/FFN, which improves
    training stability especially for deeper models.

    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
        |__________________________|   |_____________________|
                (residual)                   (residual)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Pre-norm for feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Mask for padding tokens [batch_size, seq_len]
                              True = masked (ignored), False = valid

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_out)

        # Pre-norm feedforward with residual
        x = x + self.ff(self.norm2(x))

        return x


@dataclass
class ModelConfig:
    """Configuration for DiffusionTransformer."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    vocab_size: int = 8192
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0


# Predefined model configurations
MODEL_CONFIGS = {
    "tiny": ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
    ),
    "small": ModelConfig(
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
    ),
    "medium": ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
    ),
    "large": ModelConfig(
        d_model=640,
        n_heads=10,
        n_layers=10,
        d_ff=2560,
    ),
}


class DiffusionTransformer(nn.Module):
    """
    Bidirectional Transformer for discrete diffusion language modeling.

    Key differences from autoregressive transformers:
    1. No causal masking - attention is bidirectional
    2. Conditions on diffusion timestep t ∈ [0, 1]
    3. Outputs logits for ALL positions (not just next token)

    The model learns to predict clean tokens from noisy (masked) input,
    conditioned on the noise level t.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id
        )

        # Learned position embedding
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Timestep embedding: sinusoidal -> MLP
        self.timestep_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm (for pre-norm architecture)
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Zero out padding embedding
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion transformer.

        Args:
            x: Token indices [batch_size, seq_len]
            t: Diffusion timestep [batch_size] in range [0, 1]
            attention_mask: Optional mask [batch_size, seq_len]
                           1 = valid token, 0 = padding (will be ignored)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Token embeddings
        h = self.token_embedding(x)  # [batch, seq_len, d_model]

        # Position embeddings
        positions = torch.arange(seq_len, device=device)
        h = h + self.position_embedding(positions)  # Broadcast over batch

        # Timestep embeddings (added to every position)
        t_emb = self.timestep_embedding(t)  # [batch, d_model]
        h = h + t_emb.unsqueeze(1)  # [batch, seq_len, d_model]

        # Create key_padding_mask for attention
        # nn.MultiheadAttention expects True = masked (ignored)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # Invert: 0 -> True (masked)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)

        # Final norm and projection
        h = self.final_norm(h)
        logits = self.output_proj(h)

        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    config_name: str = "small",
    vocab_size: int = 8192,
    max_seq_len: int = 256,
    pad_token_id: int = 0,
    **kwargs
) -> DiffusionTransformer:
    """
    Factory function to create a DiffusionTransformer from a named config.

    Args:
        config_name: One of "tiny", "small", "medium", "large"
        vocab_size: Vocabulary size (default: 8192)
        max_seq_len: Maximum sequence length (default: 256)
        pad_token_id: Padding token ID (default: 0)
        **kwargs: Additional config overrides

    Returns:
        Configured DiffusionTransformer model
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")

    # Start with base config
    base_config = MODEL_CONFIGS[config_name]

    # Create new config with overrides
    config = ModelConfig(
        d_model=kwargs.get("d_model", base_config.d_model),
        n_heads=kwargs.get("n_heads", base_config.n_heads),
        n_layers=kwargs.get("n_layers", base_config.n_layers),
        d_ff=kwargs.get("d_ff", base_config.d_ff),
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=kwargs.get("dropout", base_config.dropout),
        pad_token_id=pad_token_id,
    )

    model = DiffusionTransformer(config)

    param_count = model.count_parameters()
    print(f"Created {config_name} model: {param_count:,} parameters")

    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


if __name__ == "__main__":
    # Test all model configurations
    print("=" * 60)
    print("Model Configuration Tests")
    print("=" * 60)

    for name in MODEL_CONFIGS:
        print(f"\n{name.upper()} model:")
        model = create_model(name)

        # Test forward pass
        batch_size, seq_len = 4, 128
        x = torch.randint(0, 8192, (batch_size, seq_len))
        t = torch.rand(batch_size)

        with torch.no_grad():
            logits = model(x, t)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Model size: {get_model_size_mb(model):.1f} MB")

        # Verify output shape
        assert logits.shape == (batch_size, seq_len, 8192)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
