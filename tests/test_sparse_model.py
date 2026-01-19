#!/usr/bin/env python3
"""
Tests for sparse_model.py - SparseDenoiser and BilateralSparseDenoiser.

Run with: pytest tests/test_sparse_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from src.core.sparse_diffusion import SparseState, SDDConfig
from src.core.sparse_model import (
    SparseModelConfig,
    SinusoidalEmbedding,
    TransformerBlock,
    IntraPositionAttention,
    InterPositionAttention,
    BilateralSparseBlock,
    SparseDenoiser,
    BilateralSparseDenoiser,
    create_sparse_model,
    create_bilateral_sparse_model,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model_config():
    """Create a small test model config."""
    return SparseModelConfig(
        vocab_size=1000,
        embed_dim=32,
        k=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        max_seq_len=32,
        encoder_dim=64,
    )


@pytest.fixture
def sparse_state(model_config):
    """Create a test SparseState."""
    batch_size, seq_len, k = 2, 16, model_config.k
    embed_dim = model_config.embed_dim

    probs = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)
    indices = torch.randint(0, model_config.vocab_size, (batch_size, seq_len, k))
    embeds = torch.randn(batch_size, seq_len, k, embed_dim)

    return SparseState(probs=probs, embeds=embeds, indices=indices)


# =============================================================================
# Tests: SparseModelConfig
# =============================================================================

class TestSparseModelConfig:
    """Tests for SparseModelConfig dataclass."""

    def test_default_values(self):
        config = SparseModelConfig()
        assert config.vocab_size == 8192
        assert config.embed_dim == 64
        assert config.k == 8
        assert config.d_model == 768
        assert config.n_heads == 8
        assert config.n_layers == 6

    def test_custom_values(self):
        config = SparseModelConfig(
            vocab_size=5000,
            d_model=256,
            n_layers=4,
        )
        assert config.vocab_size == 5000
        assert config.d_model == 256
        assert config.n_layers == 4


# =============================================================================
# Tests: SinusoidalEmbedding
# =============================================================================

class TestSinusoidalEmbedding:
    """Tests for SinusoidalEmbedding."""

    def test_position_embedding_shape(self):
        embed = SinusoidalEmbedding(dim=64, max_len=100)
        positions = torch.arange(32)
        output = embed(positions)
        assert output.shape == (32, 64)

    def test_timestep_embedding_shape(self):
        embed = SinusoidalEmbedding(dim=64)
        timesteps = torch.rand(8)
        output = embed(timesteps)
        assert output.shape == (8, 64)

    def test_different_positions_different_embeddings(self):
        embed = SinusoidalEmbedding(dim=64)
        # Use 2D input for position mode (not timestep mode)
        pos = torch.arange(32).unsqueeze(0)  # [1, 32]
        output = embed(pos)  # [32, 64]
        # Different positions should have different embeddings
        assert not torch.allclose(output[0], output[10])
        assert not torch.allclose(output[10], output[20])


# =============================================================================
# Tests: TransformerBlock
# =============================================================================

class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_forward_without_cross_attention(self):
        block = TransformerBlock(
            d_model=64,
            n_heads=4,
            d_ff=128,
            has_cross_attention=False,
        )
        x = torch.randn(2, 16, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_forward_with_cross_attention(self):
        block = TransformerBlock(
            d_model=64,
            n_heads=4,
            d_ff=128,
            has_cross_attention=True,
        )
        x = torch.randn(2, 16, 64)
        encoder_output = torch.randn(2, 10, 64)
        output = block(x, encoder_output)
        assert output.shape == x.shape

    def test_forward_with_encoder_mask(self):
        block = TransformerBlock(
            d_model=64,
            n_heads=4,
            d_ff=128,
            has_cross_attention=True,
        )
        x = torch.randn(2, 16, 64)
        encoder_output = torch.randn(2, 10, 64)
        encoder_mask = torch.ones(2, 10)
        encoder_mask[:, 8:] = 0  # Mask last 2 positions

        output = block(x, encoder_output, encoder_mask)
        assert output.shape == x.shape


# =============================================================================
# Tests: IntraPositionAttention
# =============================================================================

class TestIntraPositionAttention:
    """Tests for IntraPositionAttention (k×k per position)."""

    def test_output_shape(self):
        attn = IntraPositionAttention(d_model=64, n_heads=4)
        h = torch.randn(2, 16, 8, 64)  # [B, L, k, D]
        probs = torch.softmax(torch.randn(2, 16, 8), dim=-1)  # [B, L, k]
        output = attn(h, probs)
        assert output.shape == h.shape

    def test_probability_bias_effect(self):
        attn = IntraPositionAttention(d_model=64, n_heads=4)
        h = torch.randn(2, 16, 8, 64)

        # Uniform probs
        probs_uniform = torch.ones(2, 16, 8) / 8
        out_uniform = attn(h, probs_uniform)

        # Peaked probs (first candidate has high prob)
        probs_peaked = torch.zeros(2, 16, 8)
        probs_peaked[:, :, 0] = 0.9
        probs_peaked[:, :, 1:] = 0.1 / 7
        out_peaked = attn(h, probs_peaked)

        # Outputs should be different
        assert not torch.allclose(out_uniform, out_peaked, atol=1e-3)


# =============================================================================
# Tests: InterPositionAttention
# =============================================================================

class TestInterPositionAttention:
    """Tests for InterPositionAttention (L×L via pooling)."""

    def test_output_shape(self):
        attn = InterPositionAttention(d_model=64, n_heads=4)
        h = torch.randn(2, 16, 8, 64)  # [B, L, k, D]
        probs = torch.softmax(torch.randn(2, 16, 8), dim=-1)  # [B, L, k]
        output = attn(h, probs)
        assert output.shape == h.shape

    def test_cross_position_information_flow(self):
        attn = InterPositionAttention(d_model=64, n_heads=4)

        # Create input where positions are clearly different
        h = torch.zeros(1, 8, 4, 64)
        h[0, 0, :, :] = 1.0  # First position different
        probs = torch.ones(1, 8, 4) / 4

        output = attn(h, probs)

        # All positions should have received some info from position 0
        # (due to attention pooling and broadcasting)
        assert not torch.allclose(output[0, 4, 0, :], torch.zeros(64))


# =============================================================================
# Tests: BilateralSparseBlock
# =============================================================================

class TestBilateralSparseBlock:
    """Tests for BilateralSparseBlock."""

    def test_forward_without_encoder(self):
        block = BilateralSparseBlock(d_model=64, n_heads=4, d_ff=128)
        h = torch.randn(2, 16, 8, 64)  # [B, L, k, D]
        probs = torch.softmax(torch.randn(2, 16, 8), dim=-1)

        output = block(h, probs)
        assert output.shape == h.shape

    def test_forward_with_encoder(self):
        block = BilateralSparseBlock(d_model=64, n_heads=4, d_ff=128)
        h = torch.randn(2, 16, 8, 64)
        probs = torch.softmax(torch.randn(2, 16, 8), dim=-1)
        encoder_output = torch.randn(2, 10, 64)

        output = block(h, probs, encoder_output)
        assert output.shape == h.shape


# =============================================================================
# Tests: SparseDenoiser (v1)
# =============================================================================

class TestSparseDenoiser:
    """Tests for SparseDenoiser (v1 - aggregation)."""

    @pytest.fixture
    def model(self, model_config):
        return SparseDenoiser(model_config)

    def test_forward_output_shape(self, model, sparse_state):
        """Test forward pass returns correct shape."""
        t = torch.rand(sparse_state.batch_size)
        logits = model(sparse_state, t)

        assert logits.shape == (
            sparse_state.batch_size,
            sparse_state.seq_len,
            model.config.vocab_size
        )

    def test_forward_with_encoder(self, model, sparse_state, model_config):
        """Test forward pass with encoder conditioning."""
        t = torch.rand(sparse_state.batch_size)
        encoder_output = torch.randn(
            sparse_state.batch_size, 10, model_config.encoder_dim
        )
        encoder_mask = torch.ones(sparse_state.batch_size, 10)

        logits = model(sparse_state, t, encoder_output, encoder_mask)

        assert logits.shape == (
            sparse_state.batch_size,
            sparse_state.seq_len,
            model.config.vocab_size
        )

    def test_denoise_step_returns_sparse_state(self, model, sparse_state):
        """Test denoise_step returns valid SparseState."""
        t = torch.rand(sparse_state.batch_size)
        new_state = model.denoise_step(sparse_state, t)

        assert isinstance(new_state, SparseState)
        assert new_state.batch_size == sparse_state.batch_size
        assert new_state.seq_len == sparse_state.seq_len
        assert new_state.k == model.config.k

        # Probs should sum to 1
        probs_sum = new_state.probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    def test_aggregate_sparse_input(self, model, sparse_state):
        """Test aggregation of sparse input."""
        aggregated = model.aggregate_sparse_input(sparse_state)

        assert aggregated.shape == (
            sparse_state.batch_size,
            sparse_state.seq_len,
            model.config.embed_dim
        )

    def test_get_embedding_table(self, model):
        """Test get_embedding_table method."""
        emb_table = model.get_embedding_table()
        assert isinstance(emb_table, nn.Embedding)
        assert emb_table.num_embeddings == model.config.vocab_size


# =============================================================================
# Tests: BilateralSparseDenoiser (v2)
# =============================================================================

class TestBilateralSparseDenoiser:
    """Tests for BilateralSparseDenoiser (v2 - attention over k)."""

    @pytest.fixture
    def model(self, model_config):
        return BilateralSparseDenoiser(model_config)

    @pytest.fixture
    def sparse_state_for_bilateral(self, model):
        """Create state using model's embedding table."""
        batch_size, seq_len, k = 2, 16, model.config.k
        probs = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)
        indices = torch.randint(0, model.config.vocab_size, (batch_size, seq_len, k))
        embeds = model.token_embedding(indices)
        return SparseState(probs=probs, embeds=embeds, indices=indices)

    def test_forward_output_shape(self, model, sparse_state_for_bilateral):
        """Test forward pass returns correct shape."""
        state = sparse_state_for_bilateral
        t = torch.rand(state.batch_size)
        logits = model(state, t)

        assert logits.shape == (
            state.batch_size,
            state.seq_len,
            model.config.vocab_size
        )

    def test_forward_with_encoder(self, model, sparse_state_for_bilateral, model_config):
        """Test forward pass with encoder conditioning."""
        state = sparse_state_for_bilateral
        t = torch.rand(state.batch_size)
        encoder_output = torch.randn(state.batch_size, 10, model_config.encoder_dim)
        encoder_mask = torch.ones(state.batch_size, 10)

        logits = model(state, t, encoder_output, encoder_mask)

        assert logits.shape == (
            state.batch_size,
            state.seq_len,
            model.config.vocab_size
        )

    def test_denoise_step_returns_sparse_state(self, model, sparse_state_for_bilateral):
        """Test denoise_step returns valid SparseState."""
        state = sparse_state_for_bilateral
        t = torch.rand(state.batch_size)
        new_state = model.denoise_step(state, t)

        assert isinstance(new_state, SparseState)
        assert new_state.batch_size == state.batch_size
        assert new_state.seq_len == state.seq_len
        assert new_state.k == model.config.k

    def test_gradient_flow(self, model, sparse_state_for_bilateral):
        """Test that gradients flow correctly."""
        state = sparse_state_for_bilateral
        t = torch.rand(state.batch_size)
        logits = model(state, t)
        loss = logits.mean()
        loss.backward()

        # Check gradients exist on key parameters
        assert model.readout_query.grad is not None
        assert model.output_proj.weight.grad is not None

    def test_parameter_count_larger_than_v1(self, model_config):
        """Test v2 has more parameters than v1 (due to bilateral attention)."""
        v1 = SparseDenoiser(model_config)
        v2 = BilateralSparseDenoiser(model_config)

        params_v1 = sum(p.numel() for p in v1.parameters())
        params_v2 = sum(p.numel() for p in v2.parameters())

        assert params_v2 > params_v1


# =============================================================================
# Tests: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for model factory functions."""

    def test_create_sparse_model(self):
        """Test create_sparse_model factory."""
        model, sdd_config = create_sparse_model(
            vocab_size=1000,
            embed_dim=32,
            k=4,
            d_model=128,
            n_layers=2,
        )

        assert isinstance(model, SparseDenoiser)
        assert isinstance(sdd_config, SDDConfig)
        assert model.config.vocab_size == 1000
        assert sdd_config.vocab_size == 1000

    def test_create_bilateral_sparse_model(self):
        """Test create_bilateral_sparse_model factory."""
        model, sdd_config = create_bilateral_sparse_model(
            vocab_size=1000,
            embed_dim=32,
            k=4,
            d_model=128,
            n_layers=2,
        )

        assert isinstance(model, BilateralSparseDenoiser)
        assert isinstance(sdd_config, SDDConfig)
        assert model.config.vocab_size == 1000

    def test_factory_without_encoder(self):
        """Test creating model without encoder (unconditional)."""
        model, _ = create_sparse_model(
            vocab_size=1000,
            encoder_dim=None,
        )

        assert model.encoder_proj is None


# =============================================================================
# Tests: Model Integration
# =============================================================================

class TestModelIntegration:
    """Integration tests for complete model functionality."""

    def test_v1_training_loop_simulation(self):
        """Simulate a training iteration with v1."""
        model, sdd_config = create_sparse_model(
            vocab_size=500,
            d_model=64,
            n_layers=1,
            encoder_dim=None,
        )

        batch_size, seq_len = 2, 8
        k = sdd_config.k

        # Create input
        probs = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)
        indices = torch.randint(0, 500, (batch_size, seq_len, k))
        embeds = model.token_embedding(indices)
        state = SparseState(probs=probs, embeds=embeds, indices=indices)

        # Forward pass
        t = torch.rand(batch_size)
        logits = model(state, t)

        # Backward pass
        loss = logits.mean()
        loss.backward()

        # Check model updated
        assert model.output_proj.weight.grad is not None

    def test_v2_training_loop_simulation(self):
        """Simulate a training iteration with v2."""
        model, sdd_config = create_bilateral_sparse_model(
            vocab_size=500,
            d_model=64,
            n_layers=1,
            encoder_dim=None,
        )

        batch_size, seq_len = 2, 8
        k = sdd_config.k

        # Create input
        probs = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)
        indices = torch.randint(0, 500, (batch_size, seq_len, k))
        embeds = model.token_embedding(indices)
        state = SparseState(probs=probs, embeds=embeds, indices=indices)

        # Forward pass
        t = torch.rand(batch_size)
        logits = model(state, t)

        # Backward pass
        loss = logits.mean()
        loss.backward()

        # Check gradients
        assert model.readout_query.grad is not None
