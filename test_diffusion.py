#!/usr/bin/env python3
"""
Comprehensive tests for diffusion.py

Run with: pytest test_diffusion.py -v
"""

import pytest
import torch
import torch.nn as nn

from model import create_model
from diffusion import DiscreteDiffusion


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def diffusion():
    """Create a diffusion process for testing."""
    return DiscreteDiffusion(
        vocab_size=1000,
        mask_token_id=3,
        pad_token_id=0,
        schedule="cosine",
    )


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    return create_model("tiny", vocab_size=1000, max_seq_len=64)


@pytest.fixture
def clean_tokens():
    """Create clean token sequences for testing."""
    batch_size, seq_len = 4, 32
    # Avoid special tokens (0-4)
    return torch.randint(5, 1000, (batch_size, seq_len))


# =============================================================================
# Tests: Noise Schedule
# =============================================================================

class TestNoiseSchedule:
    """Tests for noise schedule."""

    def test_mask_rate_at_zero(self, diffusion):
        """Test that mask rate is 0 at t=0."""
        t = torch.tensor([0.0])
        rate = diffusion.get_mask_rate(t)

        assert torch.isclose(rate, torch.tensor([0.0]), atol=1e-6)

    def test_mask_rate_at_one(self, diffusion):
        """Test that mask rate is 1 at t=1."""
        t = torch.tensor([1.0])
        rate = diffusion.get_mask_rate(t)

        assert torch.isclose(rate, torch.tensor([1.0]), atol=1e-6)

    def test_mask_rate_monotonic(self, diffusion):
        """Test that mask rate is monotonically increasing."""
        t = torch.linspace(0, 1, 100)
        rates = diffusion.get_mask_rate(t)

        # Check monotonicity
        diffs = rates[1:] - rates[:-1]
        assert (diffs >= -1e-6).all()  # Allow tiny numerical errors

    def test_cosine_schedule_smooth(self, diffusion):
        """Test that cosine schedule is smooth (no sharp jumps)."""
        t = torch.linspace(0, 1, 100)
        rates = diffusion.get_mask_rate(t)

        # Check that differences are bounded
        diffs = (rates[1:] - rates[:-1]).abs()
        assert diffs.max() < 0.1  # No jumps > 0.1

    def test_linear_schedule(self):
        """Test linear schedule."""
        diffusion = DiscreteDiffusion(schedule="linear")

        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        rates = diffusion.get_mask_rate(t)

        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        assert torch.allclose(rates, expected)

    def test_invalid_schedule_raises(self):
        """Test that invalid schedule raises error."""
        diffusion = DiscreteDiffusion(schedule="invalid")

        with pytest.raises(ValueError):
            diffusion.get_mask_rate(torch.tensor([0.5]))


# =============================================================================
# Tests: Forward Process (q_sample)
# =============================================================================

class TestForwardProcess:
    """Tests for forward process (noise injection)."""

    def test_output_shapes(self, diffusion, clean_tokens):
        """Test that outputs have correct shapes."""
        t = torch.rand(clean_tokens.shape[0])
        x_noisy, mask = diffusion.q_sample(clean_tokens, t)

        assert x_noisy.shape == clean_tokens.shape
        assert mask.shape == clean_tokens.shape

    def test_no_masking_at_t_zero(self, diffusion, clean_tokens):
        """Test that no tokens are masked at t=0."""
        t = torch.zeros(clean_tokens.shape[0])
        x_noisy, mask = diffusion.q_sample(clean_tokens, t)

        assert mask.sum() == 0
        assert torch.equal(x_noisy, clean_tokens)

    def test_full_masking_at_t_one(self, diffusion, clean_tokens):
        """Test that all tokens are masked at t=1."""
        t = torch.ones(clean_tokens.shape[0])
        x_noisy, mask = diffusion.q_sample(clean_tokens, t)

        # All non-pad tokens should be masked
        non_pad = clean_tokens != diffusion.pad_token_id
        assert mask.sum() == non_pad.sum()
        assert (x_noisy[mask] == diffusion.mask_token_id).all()

    def test_mask_rate_approximately_correct(self, diffusion):
        """Test that actual mask rate matches expected."""
        batch_size, seq_len = 100, 100
        tokens = torch.randint(5, 1000, (batch_size, seq_len))

        for t_val in [0.25, 0.5, 0.75]:
            t = torch.full((batch_size,), t_val)
            _, mask = diffusion.q_sample(tokens, t)

            actual_rate = mask.float().mean().item()
            expected_rate = diffusion.get_mask_rate(torch.tensor([t_val])).item()

            # Allow 10% relative tolerance due to randomness
            assert abs(actual_rate - expected_rate) < 0.1

    def test_padding_never_masked(self, diffusion):
        """Test that padding tokens are never masked."""
        batch_size, seq_len = 4, 32
        tokens = torch.randint(5, 1000, (batch_size, seq_len))
        tokens[:, 16:] = diffusion.pad_token_id  # Add padding

        t = torch.ones(batch_size)  # Maximum noise
        x_noisy, mask = diffusion.q_sample(tokens, t)

        # Padding positions should not be masked
        pad_positions = tokens == diffusion.pad_token_id
        assert (~mask[pad_positions]).all()
        assert (x_noisy[pad_positions] == diffusion.pad_token_id).all()

    def test_masked_positions_have_mask_token(self, diffusion, clean_tokens):
        """Test that masked positions contain MASK token."""
        t = torch.full((clean_tokens.shape[0],), 0.5)
        x_noisy, mask = diffusion.q_sample(clean_tokens, t)

        assert (x_noisy[mask] == diffusion.mask_token_id).all()

    def test_unmasked_positions_unchanged(self, diffusion, clean_tokens):
        """Test that unmasked positions retain original tokens."""
        t = torch.full((clean_tokens.shape[0],), 0.5)
        x_noisy, mask = diffusion.q_sample(clean_tokens, t)

        assert torch.equal(x_noisy[~mask], clean_tokens[~mask])


# =============================================================================
# Tests: Training Losses
# =============================================================================

class TestTrainingLosses:
    """Tests for training loss computation."""

    def test_loss_is_scalar(self, diffusion, tiny_model, clean_tokens):
        """Test that loss is a scalar."""
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens)

        assert loss.dim() == 0
        assert loss.shape == torch.Size([])

    def test_loss_is_positive(self, diffusion, tiny_model, clean_tokens):
        """Test that loss is positive."""
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens)

        assert loss.item() > 0

    def test_metrics_keys(self, diffusion, tiny_model, clean_tokens):
        """Test that metrics contain expected keys."""
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens)

        expected_keys = ["loss", "accuracy", "mask_rate", "num_masked"]
        for key in expected_keys:
            assert key in metrics

    def test_accuracy_in_range(self, diffusion, tiny_model, clean_tokens):
        """Test that accuracy is in [0, 1]."""
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens)

        assert 0 <= metrics["accuracy"] <= 1

    def test_mask_rate_in_range(self, diffusion, tiny_model, clean_tokens):
        """Test that mask rate is in [0, 1]."""
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens)

        assert 0 <= metrics["mask_rate"] <= 1

    def test_loss_with_fixed_timestep(self, diffusion, tiny_model, clean_tokens):
        """Test loss with fixed timestep."""
        t = torch.full((clean_tokens.shape[0],), 0.5)
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens, t=t)

        assert loss.item() > 0

    def test_gradient_flows(self, diffusion, tiny_model, clean_tokens):
        """Test that gradients flow through loss."""
        loss, metrics = diffusion.training_losses(tiny_model, clean_tokens)
        loss.backward()

        # Check that parameters have gradients
        has_grad = False
        for param in tiny_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad


# =============================================================================
# Tests: Reverse Process (Sampling)
# =============================================================================

class TestReverseProcess:
    """Tests for reverse process (denoising/sampling)."""

    def test_sample_output_shape(self, diffusion, tiny_model):
        """Test that sample has correct shape."""
        samples = diffusion.sample(
            tiny_model,
            batch_size=4,
            seq_len=32,
            num_steps=5,
            device="cpu"
        )

        assert samples.shape == (4, 32)

    def test_sample_output_dtype(self, diffusion, tiny_model):
        """Test that sample has correct dtype."""
        samples = diffusion.sample(
            tiny_model,
            batch_size=2,
            seq_len=16,
            num_steps=5,
            device="cpu"
        )

        assert samples.dtype == torch.long

    def test_sample_no_mask_tokens(self, diffusion, tiny_model):
        """Test that final samples have no MASK tokens."""
        samples = diffusion.sample(
            tiny_model,
            batch_size=4,
            seq_len=16,
            num_steps=20,  # More steps for complete denoising
            device="cpu"
        )

        # Should have no (or very few) mask tokens
        mask_count = (samples == diffusion.mask_token_id).sum().item()
        total_tokens = samples.numel()

        # Allow up to 5% mask tokens (edge case with low temperature)
        assert mask_count / total_tokens < 0.05

    def test_sample_with_prompt(self, diffusion, tiny_model):
        """Test sampling with a prompt."""
        prompt = torch.randint(5, 100, (2, 8))

        samples = diffusion.sample(
            tiny_model,
            batch_size=2,
            seq_len=24,
            num_steps=10,
            device="cpu",
            prompt=prompt
        )

        # Prompt should be preserved
        assert torch.equal(samples[:, :8], prompt)

    def test_sample_temperature_effect(self, diffusion, tiny_model):
        """Test that temperature affects sampling."""
        torch.manual_seed(42)
        samples_low = diffusion.sample(
            tiny_model, 2, 16, 10, temperature=0.5, device="cpu"
        )

        torch.manual_seed(42)
        samples_high = diffusion.sample(
            tiny_model, 2, 16, 10, temperature=2.0, device="cpu"
        )

        # Different temperatures should produce different results
        # (with same seed, the randomness comes from different logit scaling)
        assert not torch.equal(samples_low, samples_high)

    def test_sample_top_k(self, diffusion, tiny_model):
        """Test sampling with top-k filtering."""
        samples = diffusion.sample(
            tiny_model,
            batch_size=2,
            seq_len=16,
            num_steps=10,
            top_k=50,
            device="cpu"
        )

        assert samples.shape == (2, 16)

    def test_sample_deterministic_with_seed(self, diffusion, tiny_model):
        """Test that sampling is deterministic with same seed."""
        torch.manual_seed(123)
        samples1 = diffusion.sample(tiny_model, 2, 16, 10, device="cpu")

        torch.manual_seed(123)
        samples2 = diffusion.sample(tiny_model, 2, 16, 10, device="cpu")

        assert torch.equal(samples1, samples2)

    def test_p_sample_step(self, diffusion, tiny_model):
        """Test single denoising step."""
        x = torch.full((2, 16), diffusion.mask_token_id)
        t = torch.tensor([0.8, 0.8])
        t_next = torch.tensor([0.6, 0.6])

        x_denoised = diffusion.p_sample_step(tiny_model, x, t, t_next)

        assert x_denoised.shape == x.shape
        # Should have fewer masks than before (on average)
        orig_masks = (x == diffusion.mask_token_id).sum()
        new_masks = (x_denoised == diffusion.mask_token_id).sum()
        # Note: this is probabilistic, so we just check it's not more
        assert new_masks <= orig_masks


# =============================================================================
# Tests: Trajectory Sampling
# =============================================================================

class TestTrajectorySampling:
    """Tests for trajectory recording during sampling."""

    def test_trajectory_length(self, diffusion, tiny_model):
        """Test that trajectory has correct length."""
        num_steps = 10
        samples, trajectory = diffusion.sample_with_trajectory(
            tiny_model,
            batch_size=2,
            seq_len=16,
            num_steps=num_steps,
            device="cpu"
        )

        # Trajectory should have num_steps + 1 entries (initial + each step)
        assert len(trajectory) == num_steps + 1

    def test_trajectory_first_is_all_masks(self, diffusion, tiny_model):
        """Test that trajectory starts with all masks."""
        samples, trajectory = diffusion.sample_with_trajectory(
            tiny_model, 2, 16, 10, device="cpu"
        )

        first = trajectory[0]
        assert (first == diffusion.mask_token_id).all()

    def test_trajectory_last_matches_samples(self, diffusion, tiny_model):
        """Test that trajectory ends with final samples."""
        samples, trajectory = diffusion.sample_with_trajectory(
            tiny_model, 2, 16, 10, device="cpu"
        )

        assert torch.equal(trajectory[-1], samples)

    def test_trajectory_masks_decrease(self, diffusion, tiny_model):
        """Test that mask count decreases over trajectory."""
        samples, trajectory = diffusion.sample_with_trajectory(
            tiny_model, 4, 32, 20, device="cpu"
        )

        mask_counts = [
            (t == diffusion.mask_token_id).sum().item()
            for t in trajectory
        ]

        # Mask count should generally decrease (allow some noise)
        # Check that final has fewer masks than initial
        assert mask_counts[-1] < mask_counts[0]


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_batch_size_one(self, diffusion, tiny_model):
        """Test with batch size of 1."""
        tokens = torch.randint(5, 1000, (1, 32))
        loss, metrics = diffusion.training_losses(tiny_model, tokens)

        assert loss.item() > 0

    def test_seq_len_one(self, diffusion, tiny_model):
        """Test with sequence length of 1."""
        tokens = torch.randint(5, 1000, (4, 1))
        loss, metrics = diffusion.training_losses(tiny_model, tokens)

        assert loss.item() > 0

    def test_all_same_tokens(self, diffusion, tiny_model):
        """Test with all same tokens."""
        tokens = torch.full((4, 32), 42)
        loss, metrics = diffusion.training_losses(tiny_model, tokens)

        assert torch.isfinite(loss)

    def test_mixed_timesteps(self, diffusion, tiny_model):
        """Test with different timesteps per sample."""
        tokens = torch.randint(5, 1000, (4, 32))
        t = torch.tensor([0.1, 0.4, 0.7, 0.9])

        loss, metrics = diffusion.training_losses(tiny_model, tokens, t=t)

        assert loss.item() > 0

    def test_sample_single_step(self, diffusion, tiny_model):
        """Test sampling with single step."""
        samples = diffusion.sample(
            tiny_model, 2, 16, num_steps=1, device="cpu"
        )

        assert samples.shape == (2, 16)

    def test_sample_many_steps(self, diffusion, tiny_model):
        """Test sampling with many steps."""
        samples = diffusion.sample(
            tiny_model, 2, 16, num_steps=100, device="cpu"
        )

        assert samples.shape == (2, 16)
        # Should have very few mask tokens with many steps
        mask_fraction = (samples == diffusion.mask_token_id).float().mean()
        assert mask_fraction < 0.1


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
