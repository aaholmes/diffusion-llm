#!/usr/bin/env python3
"""
Comprehensive tests for model.py

Run with: pytest test_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from model import (
    SinusoidalPositionEmbeddings,
    TransformerBlock,
    DiffusionTransformer,
    ModelConfig,
    MODEL_CONFIGS,
    create_model,
    get_model_size_mb,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    return create_model("tiny", vocab_size=1000, max_seq_len=64)


@pytest.fixture
def batch_data():
    """Create sample batch data."""
    batch_size, seq_len = 4, 32
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    timesteps = torch.rand(batch_size)
    return tokens, timesteps


# =============================================================================
# Tests: SinusoidalPositionEmbeddings
# =============================================================================

class TestSinusoidalPositionEmbeddings:
    """Tests for timestep embeddings."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        dim = 128
        batch_size = 8
        embed = SinusoidalPositionEmbeddings(dim)

        t = torch.rand(batch_size)
        output = embed(t)

        assert output.shape == (batch_size, dim)

    def test_output_dtype(self):
        """Test that output has correct dtype."""
        embed = SinusoidalPositionEmbeddings(64)
        t = torch.rand(4)
        output = embed(t)

        assert output.dtype == torch.float32

    def test_different_timesteps_different_embeddings(self):
        """Test that different timesteps produce different embeddings."""
        embed = SinusoidalPositionEmbeddings(64)

        t1 = torch.tensor([0.2])
        t2 = torch.tensor([0.8])

        emb1 = embed(t1)
        emb2 = embed(t2)

        # Embeddings should be different
        assert not torch.allclose(emb1, emb2)

    def test_same_timestep_same_embedding(self):
        """Test that same timestep produces same embedding."""
        embed = SinusoidalPositionEmbeddings(64)

        t = torch.tensor([0.5, 0.5])
        output = embed(t)

        assert torch.allclose(output[0], output[1])

    def test_boundary_timesteps(self):
        """Test embeddings at t=0 and t=1."""
        embed = SinusoidalPositionEmbeddings(64)

        t = torch.tensor([0.0, 1.0])
        output = embed(t)

        # Should not be NaN or Inf
        assert torch.isfinite(output).all()

        # t=0 and t=1 should produce different embeddings
        assert not torch.allclose(output[0], output[1])


# =============================================================================
# Tests: TransformerBlock
# =============================================================================

class TestTransformerBlock:
    """Tests for transformer block."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        d_model, n_heads, d_ff = 64, 4, 256
        block = TransformerBlock(d_model, n_heads, d_ff)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that block includes residual connections."""
        block = TransformerBlock(64, 4, 256)

        # With zero input, output should be close to zero
        # (residual + near-zero attention/ffn output)
        x = torch.zeros(2, 16, 64)
        output = block(x)

        # Output should be small (residual of zero + small initialized values)
        assert output.abs().mean() < 1.0

    def test_with_padding_mask(self):
        """Test that padding mask is respected."""
        block = TransformerBlock(64, 4, 256)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 64)

        # Create padding mask (True = ignored)
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        key_padding_mask[:, 8:] = True  # Last 8 positions are padding

        # Should not crash with mask
        output = block(x, key_padding_mask=key_padding_mask)
        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = TransformerBlock(64, 4, 256)

        x = torch.randn(2, 16, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


# =============================================================================
# Tests: DiffusionTransformer
# =============================================================================

class TestDiffusionTransformer:
    """Tests for the main transformer model."""

    def test_output_shape(self, tiny_model, batch_data):
        """Test that output has correct shape."""
        tokens, timesteps = batch_data
        tokens = tokens[:, :64]  # Match max_seq_len

        output = tiny_model(tokens, timesteps)

        assert output.shape == (tokens.shape[0], tokens.shape[1], 1000)

    def test_output_dtype(self, tiny_model, batch_data):
        """Test that output has correct dtype."""
        tokens, timesteps = batch_data
        tokens = tokens[:, :64]

        output = tiny_model(tokens, timesteps)

        assert output.dtype == torch.float32

    def test_different_timesteps_different_outputs(self, tiny_model):
        """Test that different timesteps produce different outputs."""
        tokens = torch.randint(0, 1000, (2, 32))

        t1 = torch.tensor([0.2, 0.2])
        t2 = torch.tensor([0.8, 0.8])

        out1 = tiny_model(tokens, t1)
        out2 = tiny_model(tokens, t2)

        assert not torch.allclose(out1, out2)

    def test_padding_handled(self, tiny_model):
        """Test that padding is handled correctly."""
        batch_size, seq_len = 2, 32
        tokens = torch.randint(1, 1000, (batch_size, seq_len))
        tokens[:, 16:] = 0  # Add padding

        timesteps = torch.rand(batch_size)
        attention_mask = (tokens != 0).float()

        # Should not crash
        output = tiny_model(tokens, timesteps, attention_mask=attention_mask)
        assert output.shape == (batch_size, seq_len, 1000)

    def test_gradient_flow(self, tiny_model):
        """Test that gradients flow through the model."""
        tokens = torch.randint(0, 1000, (2, 32))
        timesteps = torch.rand(2)

        # Enable gradient tracking
        output = tiny_model(tokens, timesteps)
        loss = output.sum()
        loss.backward()

        # Check that parameters have gradients
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self, tiny_model):
        """Test model in eval mode."""
        tiny_model.eval()

        tokens = torch.randint(0, 1000, (2, 32))
        timesteps = torch.rand(2)

        with torch.no_grad():
            output = tiny_model(tokens, timesteps)

        assert output.shape == (2, 32, 1000)

    def test_parameter_count(self, tiny_model):
        """Test parameter counting."""
        count = tiny_model.count_parameters()

        assert count > 0
        assert isinstance(count, int)

    def test_config_stored(self, tiny_model):
        """Test that config is stored on model."""
        assert hasattr(tiny_model, 'config')
        assert tiny_model.config.vocab_size == 1000
        assert tiny_model.config.max_seq_len == 64


# =============================================================================
# Tests: Model Configurations
# =============================================================================

class TestModelConfigurations:
    """Tests for model configurations."""

    def test_all_configs_exist(self):
        """Test that all expected configs exist."""
        expected = ["tiny", "small", "medium", "large"]
        for name in expected:
            assert name in MODEL_CONFIGS

    def test_configs_increasing_size(self):
        """Test that configs have increasing parameter counts."""
        sizes = []
        for name in ["tiny", "small", "medium", "large"]:
            model = create_model(name)
            sizes.append(model.count_parameters())

        # Each model should be larger than the previous
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i-1]

    def test_create_model_with_custom_vocab(self):
        """Test creating model with custom vocab size."""
        model = create_model("tiny", vocab_size=5000)

        tokens = torch.randint(0, 5000, (2, 32))
        timesteps = torch.rand(2)
        output = model(tokens, timesteps)

        assert output.shape[-1] == 5000

    def test_create_model_with_custom_seq_len(self):
        """Test creating model with custom sequence length."""
        model = create_model("tiny", max_seq_len=128)

        tokens = torch.randint(0, 8192, (2, 128))
        timesteps = torch.rand(2)
        output = model(tokens, timesteps)

        assert output.shape[1] == 128

    def test_invalid_config_raises(self):
        """Test that invalid config name raises error."""
        with pytest.raises(ValueError):
            create_model("invalid_config")


# =============================================================================
# Tests: Weight Initialization
# =============================================================================

class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_embedding_padding_zero(self, tiny_model):
        """Test that padding embedding is zero."""
        pad_id = tiny_model.pad_token_id
        pad_embedding = tiny_model.token_embedding.weight[pad_id]

        assert torch.allclose(pad_embedding, torch.zeros_like(pad_embedding))

    def test_weights_not_all_zero(self, tiny_model):
        """Test that weights are not all zero."""
        for name, param in tiny_model.named_parameters():
            if 'weight' in name and param.numel() > 1:
                # Check that not all values are zero
                assert param.abs().sum() > 0, f"{name} is all zeros"

    def test_weights_reasonable_scale(self, tiny_model):
        """Test that weights have reasonable scale."""
        for name, param in tiny_model.named_parameters():
            if 'weight' in name:
                # Skip LayerNorm weights (initialized to 1.0, so std=0)
                if 'norm' in name:
                    continue
                std = param.std().item()
                # Should be in reasonable range (initialized with std=0.02)
                assert std < 1.0, f"{name} has std={std}, too large"
                assert std > 0.001 or param.numel() == 1, f"{name} has std={std}, too small"


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_batch_size_one(self, tiny_model):
        """Test with batch size of 1."""
        tokens = torch.randint(0, 1000, (1, 32))
        timesteps = torch.rand(1)

        output = tiny_model(tokens, timesteps)
        assert output.shape == (1, 32, 1000)

    def test_seq_len_one(self, tiny_model):
        """Test with sequence length of 1."""
        tokens = torch.randint(0, 1000, (4, 1))
        timesteps = torch.rand(4)

        output = tiny_model(tokens, timesteps)
        assert output.shape == (4, 1, 1000)

    def test_all_same_token(self, tiny_model):
        """Test with all same tokens."""
        tokens = torch.full((4, 32), 42)
        timesteps = torch.rand(4)

        output = tiny_model(tokens, timesteps)
        assert output.shape == (4, 32, 1000)
        assert torch.isfinite(output).all()

    def test_all_padding(self, tiny_model):
        """Test with all padding tokens."""
        tokens = torch.zeros(4, 32, dtype=torch.long)
        timesteps = torch.rand(4)

        output = tiny_model(tokens, timesteps)
        assert output.shape == (4, 32, 1000)
        assert torch.isfinite(output).all()

    def test_timestep_zero(self, tiny_model):
        """Test with timestep = 0."""
        tokens = torch.randint(0, 1000, (2, 32))
        timesteps = torch.zeros(2)

        output = tiny_model(tokens, timesteps)
        assert torch.isfinite(output).all()

    def test_timestep_one(self, tiny_model):
        """Test with timestep = 1."""
        tokens = torch.randint(0, 1000, (2, 32))
        timesteps = torch.ones(2)

        output = tiny_model(tokens, timesteps)
        assert torch.isfinite(output).all()


# =============================================================================
# Tests: Utilities
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_get_model_size_mb(self, tiny_model):
        """Test model size calculation."""
        size_mb = get_model_size_mb(tiny_model)

        assert size_mb > 0
        assert isinstance(size_mb, float)

    def test_model_size_increases_with_config(self):
        """Test that model size increases with larger configs."""
        tiny = create_model("tiny")
        small = create_model("small")

        tiny_size = get_model_size_mb(tiny)
        small_size = get_model_size_mb(small)

        assert small_size > tiny_size


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
