#!/usr/bin/env python3
"""
Diffusion Transformer model for discrete diffusion language modeling.

Architecture:
- Bidirectional transformer (no causal masking)
- Timestep conditioning via sinusoidal embeddings
- Pre-norm transformer blocks
- Optional cross-attention for conditional generation
- Output projection to vocabulary logits

Usage:
    from src.core.model import create_model, create_conditional_model

    # Unconditional model
    model = create_model("small")  # 25M params
    logits = model(tokens, timestep)  # [batch, seq_len, vocab_size]

    # Conditional model (encoder-decoder)
    model = create_conditional_model("small", "small")
    logits = model(decoder_tokens, timestep, encoder_tokens)
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

    Maps scalar timestep t in [0, 1] to a vector of dimension `dim`.
    Uses the same formulation as the original Transformer positional encoding,
    but applied to the continuous timestep value.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        freq = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        args = t[:, None] * freq[None, :] * 1000
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embeddings


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture and optional cross-attention.

    Architecture (without cross-attention):
        x -> LayerNorm -> SelfAttn -> + -> LayerNorm -> FFN -> +

    Architecture (with cross-attention):
        x -> LayerNorm -> SelfAttn -> + -> LayerNorm -> CrossAttn -> + -> LayerNorm -> FFN -> +
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        has_cross_attention: bool = False,
    ):
        super().__init__()
        self.has_cross_attention = has_cross_attention

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (optional)
        if has_cross_attention:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.dropout_cross = nn.Dropout(dropout)

        # Feedforward
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
        encoder_output: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_out)

        # Cross-attention
        if self.has_cross_attention and encoder_output is not None:
            normed = self.norm_cross(x)
            cross_out, _ = self.cross_attn(
                normed, encoder_output, encoder_output,
                key_padding_mask=encoder_padding_mask,
                need_weights=False
            )
            x = x + self.dropout_cross(cross_out)

        # Feedforward
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
    has_cross_attention: bool = False


# Predefined model configurations
# Jetson latency estimates assume FP16, 15 diffusion steps
MODEL_CONFIGS = {
    "tiny": ModelConfig(      # ~7.6M params, ~0.2s on Jetson
        d_model=256, n_heads=4, n_layers=4, d_ff=1024,
    ),
    "small": ModelConfig(     # ~17M params, ~0.4s on Jetson
        d_model=384, n_heads=6, n_layers=6, d_ff=1536,
    ),
    "medium": ModelConfig(    # ~34M params, ~0.6s on Jetson
        d_model=512, n_heads=8, n_layers=8, d_ff=2048,
    ),
    "large": ModelConfig(     # ~61M params, ~0.8s on Jetson
        d_model=640, n_heads=10, n_layers=10, d_ff=2560,
    ),
    "xlarge": ModelConfig(    # ~110M params, ~1.2s on Jetson - recommended
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
    ),
    "xxlarge": ModelConfig(   # ~250M params, ~2.0s on Jetson - maximum
        d_model=1024, n_heads=16, n_layers=16, d_ff=4096,
    ),
}


class DiffusionTransformer(nn.Module):
    """
    Bidirectional Transformer for discrete diffusion language modeling.

    Key differences from autoregressive transformers:
    1. No causal masking - attention is bidirectional
    2. Conditions on diffusion timestep t in [0, 1]
    3. Outputs logits for ALL positions (not just next token)
    4. Optional cross-attention for conditional generation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

        # Position embedding
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, config.has_cross_attention,
            )
            for _ in range(config.n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device

        # Embeddings
        h = self.token_embedding(x)
        positions = torch.arange(seq_len, device=device)
        h = h + self.position_embedding(positions)
        t_emb = self.timestep_embedding(t)
        h = h + t_emb.unsqueeze(1)

        # Masks
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        encoder_padding_mask = None
        if encoder_attention_mask is not None:
            encoder_padding_mask = encoder_attention_mask == 0

        # Transformer blocks
        for block in self.blocks:
            h = block(h, encoder_output, key_padding_mask, encoder_padding_mask)

        # Output
        h = self.final_norm(h)
        logits = self.output_proj(h)

        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class TextEncoder(nn.Module):
    """
    Bidirectional transformer encoder for conditioning.

    Encodes prompt/condition text into context vectors that the decoder
    can attend to via cross-attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, has_cross_attention=False,
            )
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device

        h = self.token_embedding(x)
        positions = torch.arange(seq_len, device=device)
        h = h + self.position_embedding(positions)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)

        h = self.final_norm(h)
        return h

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class ConditionalDiffusionLM(nn.Module):
    """
    Complete encoder-decoder model for conditional text generation.

    Training stages (compute-efficient approach):
    1. Train decoder unconditionally (no encoder)
    2. Freeze decoder, train encoder + cross-attention only
    3. Optional: Fine-tune everything with low learning rate
    """

    def __init__(self, encoder_config: ModelConfig, decoder_config: ModelConfig):
        super().__init__()

        if not decoder_config.has_cross_attention:
            raise ValueError("Decoder config must have has_cross_attention=True")

        if encoder_config.d_model != decoder_config.d_model:
            raise ValueError(
                f"Encoder d_model ({encoder_config.d_model}) must match "
                f"decoder d_model ({decoder_config.d_model})"
            )

        self.encoder = TextEncoder(encoder_config)
        self.decoder = DiffusionTransformer(decoder_config)

    def forward(
        self,
        decoder_input: torch.Tensor,
        t: torch.Tensor,
        encoder_input: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_output = self.encoder(encoder_input, encoder_attention_mask)
        logits = self.decoder(
            decoder_input, t,
            attention_mask=decoder_attention_mask,
            encoder_output=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
        )
        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_decoder(self):
        """Freeze decoder parameters (for stage 2 training)."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        # Unfreeze cross-attention layers
        for block in self.decoder.blocks:
            if block.has_cross_attention:
                for param in block.cross_attn.parameters():
                    param.requires_grad = True
                for param in block.norm_cross.parameters():
                    param.requires_grad = True
                for param in block.dropout_cross.parameters():
                    param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters (for stage 3 fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True


def create_model(
    config_name: str = "small",
    vocab_size: int = 8192,
    max_seq_len: int = 256,
    pad_token_id: int = 0,
    **kwargs
) -> DiffusionTransformer:
    """Factory function to create a DiffusionTransformer from a named config."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")

    base_config = MODEL_CONFIGS[config_name]

    config = ModelConfig(
        d_model=kwargs.get("d_model", base_config.d_model),
        n_heads=kwargs.get("n_heads", base_config.n_heads),
        n_layers=kwargs.get("n_layers", base_config.n_layers),
        d_ff=kwargs.get("d_ff", base_config.d_ff),
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=kwargs.get("dropout", base_config.dropout),
        pad_token_id=pad_token_id,
        has_cross_attention=kwargs.get("has_cross_attention", False),
    )

    model = DiffusionTransformer(config)
    param_count = model.count_parameters()
    print(f"Created {config_name} model: {param_count:,} parameters")

    return model


def create_encoder(
    config_name: str = "small",
    vocab_size: int = 8192,
    max_seq_len: int = 256,
    pad_token_id: int = 0,
    **kwargs
) -> TextEncoder:
    """Factory function to create a TextEncoder from a named config."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")

    base_config = MODEL_CONFIGS[config_name]

    config = ModelConfig(
        d_model=kwargs.get("d_model", base_config.d_model),
        n_heads=kwargs.get("n_heads", base_config.n_heads),
        n_layers=kwargs.get("n_layers", base_config.n_layers),
        d_ff=kwargs.get("d_ff", base_config.d_ff),
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=kwargs.get("dropout", base_config.dropout),
        pad_token_id=pad_token_id,
        has_cross_attention=False,
    )

    model = TextEncoder(config)
    param_count = model.count_parameters()
    print(f"Created {config_name} encoder: {param_count:,} parameters")

    return model


def create_conditional_model(
    encoder_config_name: str = "small",
    decoder_config_name: str = "small",
    vocab_size: int = 8192,
    encoder_max_seq_len: int = 64,
    decoder_max_seq_len: int = 192,
    pad_token_id: int = 0,
    **kwargs
) -> ConditionalDiffusionLM:
    """Factory function to create a ConditionalDiffusionLM."""
    if encoder_config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder config: {encoder_config_name}")
    if decoder_config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown decoder config: {decoder_config_name}")

    encoder_base = MODEL_CONFIGS[encoder_config_name]
    decoder_base = MODEL_CONFIGS[decoder_config_name]

    d_model = kwargs.get("d_model", decoder_base.d_model)

    encoder_config = ModelConfig(
        d_model=d_model,
        n_heads=kwargs.get("encoder_n_heads", encoder_base.n_heads),
        n_layers=kwargs.get("encoder_n_layers", encoder_base.n_layers),
        d_ff=kwargs.get("encoder_d_ff", encoder_base.d_ff),
        vocab_size=vocab_size,
        max_seq_len=encoder_max_seq_len,
        dropout=kwargs.get("dropout", encoder_base.dropout),
        pad_token_id=pad_token_id,
        has_cross_attention=False,
    )

    decoder_config = ModelConfig(
        d_model=d_model,
        n_heads=kwargs.get("decoder_n_heads", decoder_base.n_heads),
        n_layers=kwargs.get("decoder_n_layers", decoder_base.n_layers),
        d_ff=kwargs.get("decoder_d_ff", decoder_base.d_ff),
        vocab_size=vocab_size,
        max_seq_len=decoder_max_seq_len,
        dropout=kwargs.get("dropout", decoder_base.dropout),
        pad_token_id=pad_token_id,
        has_cross_attention=True,
    )

    model = ConditionalDiffusionLM(encoder_config, decoder_config)
    param_count = model.count_parameters()
    print(f"Created conditional model (encoder={encoder_config_name}, "
          f"decoder={decoder_config_name}): {param_count:,} parameters")

    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


if __name__ == "__main__":
    print("=" * 60)
    print("Model Configuration Tests")
    print("=" * 60)

    for name in MODEL_CONFIGS:
        print(f"\n{name.upper()} model:")
        model = create_model(name)

        batch_size, seq_len = 4, 128
        x = torch.randint(0, 8192, (batch_size, seq_len))
        t = torch.rand(batch_size)

        with torch.no_grad():
            logits = model(x, t)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Model size: {get_model_size_mb(model):.1f} MB")

        assert logits.shape == (batch_size, seq_len, 8192)

    print("\n" + "=" * 60)
    print("Conditional Model Test")
    print("=" * 60)

    model = create_conditional_model("tiny", "tiny", vocab_size=256)
    enc_input = torch.randint(0, 256, (2, 32))
    dec_input = torch.randint(0, 256, (2, 64))
    t = torch.rand(2)

    with torch.no_grad():
        logits = model(dec_input, t, enc_input)

    print(f"  Encoder input: {enc_input.shape}")
    print(f"  Decoder input: {dec_input.shape}")
    print(f"  Output shape: {logits.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
