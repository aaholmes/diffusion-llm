#!/usr/bin/env python3
"""
Tests for sparse_diffusion.py - SparseState and SparseDiffusion components.

Run with: pytest tests/test_sparse_diffusion.py -v
"""

import pytest
import torch
import torch.nn as nn

from src.core.sparse_diffusion import SDDConfig, SparseState, SparseDiffusion


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sdd_config():
    """Create a test SDDConfig."""
    return SDDConfig(
        vocab_size=1000,
        embed_dim=32,
        k=4,
        pad_token_id=0,
        ignore_token_id=5,
    )


@pytest.fixture
def embedding_table(sdd_config):
    """Create a test embedding table."""
    return nn.Embedding(sdd_config.vocab_size, sdd_config.embed_dim)


@pytest.fixture
def diffusion(sdd_config, embedding_table):
    """Create a SparseDiffusion instance for testing."""
    return SparseDiffusion(sdd_config, embedding_table)


@pytest.fixture
def sparse_state(sdd_config, embedding_table):
    """Create a test SparseState."""
    batch_size, seq_len, k = 2, 8, sdd_config.k
    probs = torch.softmax(torch.randn(batch_size, seq_len, k), dim=-1)
    indices = torch.randint(0, sdd_config.vocab_size, (batch_size, seq_len, k))
    return SparseState(probs=probs, indices=indices)


# =============================================================================
# Tests: SDDConfig
# =============================================================================

class TestSDDConfig:
    """Tests for SDDConfig dataclass."""

    def test_default_values(self):
        config = SDDConfig()
        assert config.vocab_size == 8192
        assert config.embed_dim == 64
        assert config.k == 8
        assert config.schedule_type == "cosine"

    def test_custom_values(self):
        config = SDDConfig(
            vocab_size=5000,
            embed_dim=128,
            k=16,
            schedule_type="linear",
        )
        assert config.vocab_size == 5000
        assert config.embed_dim == 128
        assert config.k == 16
        assert config.schedule_type == "linear"


# =============================================================================
# Tests: SparseState
# =============================================================================

class TestSparseState:
    """Tests for SparseState class."""

    def test_properties(self, sparse_state):
        """Test SparseState properties."""
        assert sparse_state.batch_size == 2
        assert sparse_state.seq_len == 8
        assert sparse_state.k == 4

    def test_top1_tokens(self, sparse_state):
        """Test top1_tokens method."""
        top1 = sparse_state.top1_tokens()
        assert top1.shape == (2, 8)
        assert (top1 == sparse_state.indices[:, :, 0]).all()

    def test_top1_probs(self, sparse_state):
        """Test top1_probs method."""
        top1_probs = sparse_state.top1_probs()
        assert top1_probs.shape == (2, 8)
        assert (top1_probs == sparse_state.probs[:, :, 0]).all()

    def test_to_device(self, sparse_state):
        """Test moving state to device."""
        device = torch.device("cpu")
        moved_state = sparse_state.to(device)
        assert moved_state.probs.device == device
        assert moved_state.indices.device == device

    def test_probs_sum_to_one(self, sparse_state):
        """Test that probabilities sum to approximately 1."""
        probs_sum = sparse_state.probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    def test_device_property(self, sparse_state):
        """Test device property returns correct device."""
        device = sparse_state.device
        assert device == sparse_state.probs.device
        assert isinstance(device, torch.device)


# =============================================================================
# Tests: SparseDiffusion
# =============================================================================

class TestSparseDiffusion:
    """Tests for SparseDiffusion class."""

    def test_get_noise_level_cosine(self, sdd_config, embedding_table):
        """Test cosine noise schedule."""
        sdd_config.schedule_type = "cosine"
        diffusion = SparseDiffusion(sdd_config, embedding_table)

        t = torch.tensor([0.0, 0.5, 1.0])
        noise = diffusion.get_noise_level(t)

        # At t=0, noise should be 0
        assert torch.isclose(noise[0], torch.tensor(0.0), atol=1e-5)
        # At t=1, noise should be 1
        assert torch.isclose(noise[2], torch.tensor(1.0), atol=1e-5)
        # At t=0.5, noise should be ~0.293 for quarter-cosine schedule
        # Formula: 1 - cos(t * pi / 2) = 1 - cos(pi/4) ≈ 0.293
        expected_mid = 1 - torch.cos(torch.tensor(0.5 * 3.14159 / 2))
        assert torch.isclose(noise[1], expected_mid, atol=1e-3)

    def test_get_noise_level_linear(self, sdd_config, embedding_table):
        """Test linear noise schedule."""
        sdd_config.schedule_type = "linear"
        diffusion = SparseDiffusion(sdd_config, embedding_table)

        t = torch.tensor([0.0, 0.5, 1.0])
        noise = diffusion.get_noise_level(t)

        # Linear schedule: noise = t
        assert torch.allclose(noise, t)

    def test_initialize_state(self, diffusion):
        """Test initializing a random state."""
        batch_size, seq_len = 4, 16
        state = diffusion.initialize_state(batch_size, seq_len, torch.device("cpu"))

        assert state.batch_size == batch_size
        assert state.seq_len == seq_len
        assert state.k == diffusion.config.k

        # Uniform probabilities: each token has 1/vocab_size probability
        # (coarse-grained interpretation - not normalized to sum to 1 over k)
        expected_prob = 1.0 / diffusion.config.vocab_size
        assert torch.allclose(state.probs, torch.full_like(state.probs, expected_prob))

        # Probs should sum to k/vocab_size, NOT 1.0
        probs_sum = state.probs.sum(dim=-1)
        expected_sum = diffusion.config.k / diffusion.config.vocab_size
        assert torch.allclose(probs_sum, torch.full_like(probs_sum, expected_sum))

    def test_add_noise_preserves_shape(self, diffusion, sparse_state):
        """Test that add_noise preserves shapes."""
        t = torch.tensor([0.5] * sparse_state.batch_size)
        noisy_state = diffusion.add_noise(sparse_state, t)

        assert noisy_state.probs.shape == sparse_state.probs.shape
        assert noisy_state.indices.shape == sparse_state.indices.shape

    def test_add_noise_at_zero(self, diffusion, sparse_state):
        """Test that noise at t=0 is minimal."""
        t = torch.tensor([0.0] * sparse_state.batch_size)
        noisy_state = diffusion.add_noise(sparse_state, t)

        # At t=0, no swaps should occur and probs should be unchanged
        assert torch.allclose(noisy_state.probs, sparse_state.probs, atol=1e-5)
        assert torch.equal(noisy_state.indices, sparse_state.indices)

    def test_add_noise_at_one(self, diffusion, sparse_state):
        """Test that noise at t=1 gives uniform probs."""
        t = torch.tensor([1.0] * sparse_state.batch_size)
        noisy_state = diffusion.add_noise(sparse_state, t)

        # At t=1, all probs should be close to uniform (1/vocab_size per token)
        # (both due to prob flattening and swapped tokens getting 1/vocab_size)
        expected_prob = 1.0 / diffusion.config.vocab_size
        assert torch.allclose(
            noisy_state.probs,
            torch.full_like(noisy_state.probs, expected_prob),
            atol=1e-5
        )

    def test_add_noise_swapping_increases_with_t(self, diffusion):
        """Test that swapping rate increases with noise level."""
        # Create clean state from tokens
        tokens = torch.randint(5, diffusion.config.vocab_size, (8, 16))
        clean_state = diffusion.state_from_tokens(tokens)

        swap_rates = []
        for t_val in [0.0, 0.3, 0.7, 1.0]:
            t = torch.tensor([t_val] * 8)
            noisy_state = diffusion.add_noise(clean_state, t)
            # Count how many indices changed
            swapped = (noisy_state.indices != clean_state.indices).float().mean()
            swap_rates.append(swapped.item())

        # Swap rate should increase monotonically with t
        for i in range(len(swap_rates) - 1):
            assert swap_rates[i] <= swap_rates[i + 1] + 0.1  # Allow small tolerance

    def test_add_noise_high_prob_swapped_less(self, diffusion):
        """Test that high-probability candidates are swapped less often."""
        # Create clean state with peaked distribution
        tokens = torch.randint(5, diffusion.config.vocab_size, (16, 32))
        clean_state = diffusion.state_from_tokens(tokens, temperature=0.01)

        # Run multiple times at intermediate noise level
        t = torch.tensor([0.5] * 16)
        top1_swap_count = 0
        other_swap_count = 0
        n_trials = 20

        for _ in range(n_trials):
            noisy_state = diffusion.add_noise(clean_state, t)
            # Check if top-1 candidate (index 0) was swapped
            top1_swapped = (noisy_state.indices[:, :, 0] != clean_state.indices[:, :, 0]).float()
            other_swapped = (noisy_state.indices[:, :, 1:] != clean_state.indices[:, :, 1:]).float()
            top1_swap_count += top1_swapped.mean().item()
            other_swap_count += other_swapped.mean().item()

        top1_swap_rate = top1_swap_count / n_trials
        other_swap_rate = other_swap_count / n_trials

        # High-prob candidates (top-1) should be swapped less than low-prob ones
        assert top1_swap_rate < other_swap_rate

    def test_add_noise_swapped_get_uniform_prob(self, diffusion):
        """Test that swapped tokens get 1/vocab_size probability."""
        tokens = torch.randint(5, diffusion.config.vocab_size, (4, 16))
        clean_state = diffusion.state_from_tokens(tokens, temperature=0.01)

        # At high noise, most things will be swapped
        t = torch.tensor([0.9] * 4)
        noisy_state = diffusion.add_noise(clean_state, t)

        # Find positions where swapping occurred
        swap_mask = (noisy_state.indices != clean_state.indices)
        if swap_mask.any():
            expected_prob = 1.0 / diffusion.config.vocab_size
            swapped_probs = noisy_state.probs[swap_mask]
            assert torch.allclose(
                swapped_probs,
                torch.full_like(swapped_probs, expected_prob),
                atol=1e-5
            )

    def test_state_from_tokens(self, diffusion):
        """Test creating state from discrete tokens."""
        tokens = torch.randint(0, diffusion.config.vocab_size, (4, 16))
        state = diffusion.state_from_tokens(tokens)

        # Ground truth should be first
        assert (state.indices[:, :, 0] == tokens).all()

        # First prob should be highest
        assert (state.probs[:, :, 0] > state.probs[:, :, 1]).all()

    def test_state_from_tokens_peaked_distribution(self, diffusion):
        """Test that state_from_tokens creates peaked distribution."""
        tokens = torch.randint(0, diffusion.config.vocab_size, (4, 16))
        state = diffusion.state_from_tokens(tokens, temperature=0.01)

        # Most probability mass should be on first token
        assert state.probs[:, :, 0].mean() > 0.9


# =============================================================================
# Tests: SparseDiffusion Training
# =============================================================================

class TestSparseDiffusionTraining:
    """Tests for SparseDiffusion training functionality."""

    @pytest.fixture
    def mock_model(self, sdd_config):
        """Create a simple mock model that returns random logits."""
        class MockModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size
                # Add a parameter to enable gradient computation
                self.dummy_param = nn.Parameter(torch.zeros(1))

            def forward(self, state, t, encoder_output=None, encoder_mask=None):
                # Return random logits that require grad through dummy_param
                logits = torch.randn(state.batch_size, state.seq_len, self.vocab_size)
                return logits + self.dummy_param * 0  # Add dummy to connect gradients

        return MockModel(sdd_config.vocab_size)

    def test_training_step_returns_loss_and_metrics(self, diffusion, mock_model):
        """Test that training_step returns valid loss and metrics."""
        tokens = torch.randint(5, diffusion.config.vocab_size, (4, 16))

        loss, metrics = diffusion.training_step(mock_model, tokens)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mean_t" in metrics

    def test_training_step_handles_padding(self, diffusion, mock_model):
        """Test that padding tokens are handled correctly."""
        tokens = torch.randint(5, diffusion.config.vocab_size, (4, 16))
        # Add padding at the end
        tokens[:, 12:] = diffusion.config.pad_token_id

        loss, metrics = diffusion.training_step(mock_model, tokens)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


# =============================================================================
# Tests: SparseDiffusion Sampling
# =============================================================================

class TestSparseDiffusionSampling:
    """Tests for SparseDiffusion sampling functionality."""

    @pytest.fixture
    def mock_denoiser(self, sdd_config, embedding_table):
        """Create a mock denoiser with denoise_step method."""
        class MockDenoiser(nn.Module):
            def __init__(self, config, emb_table):
                super().__init__()
                self.config = config
                self.token_embedding = emb_table

            def eval(self):
                return self

            def denoise_step(self, state, t, encoder_output=None, encoder_mask=None,
                           temperature=1.0):
                # Return random new state
                B, L, k = state.probs.shape
                new_probs = torch.softmax(torch.randn(B, L, k), dim=-1)
                new_indices = torch.randint(0, self.config.vocab_size, (B, L, k))
                return SparseState(probs=new_probs, indices=new_indices)

        return MockDenoiser(sdd_config, embedding_table)

    def test_sample_returns_tokens(self, diffusion, mock_denoiser):
        """Test that sample returns discrete tokens."""
        batch_size, seq_len = 2, 8
        tokens, final_state = diffusion.sample(
            mock_denoiser,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=5,
            device="cpu",
        )

        assert tokens.shape == (batch_size, seq_len)
        assert tokens.dtype == torch.long
        assert final_state.batch_size == batch_size
        assert final_state.seq_len == seq_len

    def test_sample_with_conditioning(self, diffusion, mock_denoiser):
        """Test sampling with encoder conditioning."""
        batch_size, seq_len = 2, 8
        encoder_output = torch.randn(batch_size, 10, 768)
        encoder_mask = torch.ones(batch_size, 10)

        tokens, final_state = diffusion.sample(
            mock_denoiser,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=5,
            encoder_output=encoder_output,
            encoder_mask=encoder_mask,
            device="cpu",
        )

        assert tokens.shape == (batch_size, seq_len)


# =============================================================================
# Tests: SparseDiffusion Decoding
# =============================================================================

class TestSparseDiffusionDecoding:
    """Tests for SparseDiffusion decoding functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        class MockTokenizer:
            def decode(self, token_ids):
                # Simple mock that just joins token IDs as strings
                return " ".join(str(t) for t in token_ids)
        return MockTokenizer()

    def test_decode_with_ignore_filters_special_tokens(self, diffusion, mock_tokenizer):
        """Test that decode_with_ignore filters out special tokens."""
        # Create tokens with some special tokens mixed in
        tokens = torch.tensor([
            [1, 10, 20, 0, 5, 30, 2],  # BOS=1, content, PAD=0, IGNORE=5, content, EOS=2
            [1, 15, 5, 5, 25, 0, 2],   # BOS=1, content, IGNORE, IGNORE, content, PAD=0, EOS=2
        ])

        results = diffusion.decode_with_ignore(tokens, mock_tokenizer)

        assert len(results) == 2
        # Should only have content tokens (10, 20, 30) and (15, 25)
        assert "10" in results[0]
        assert "20" in results[0]
        assert "30" in results[0]
        assert "15" in results[1]
        assert "25" in results[1]
        # Should NOT have special tokens
        for result in results:
            assert "0" not in result.split()  # PAD
            assert "1" not in result.split()  # BOS
            assert "2" not in result.split()  # EOS
            assert "5" not in result.split()  # IGNORE

    def test_decode_with_ignore_handles_empty_sequence(self, diffusion, mock_tokenizer):
        """Test decode_with_ignore with sequence of only special tokens."""
        # Only special tokens
        tokens = torch.tensor([
            [1, 0, 0, 5, 5, 2],  # BOS, PAD, PAD, IGNORE, IGNORE, EOS
        ])

        results = diffusion.decode_with_ignore(tokens, mock_tokenizer)

        assert len(results) == 1
        # Should be empty or whitespace only
        assert results[0].strip() == ""
