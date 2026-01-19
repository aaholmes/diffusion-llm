#!/usr/bin/env python3
"""
Integration tests for the complete diffusion LM pipeline.

These tests verify that all components work together correctly.

Run with: pytest test_integration.py -v
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.data_prep import (
    DataConfig,
    load_stories,
    train_tokenizer,
    tokenize_dataset,
    compute_statistics,
    main as data_prep_main,
)
from src.core.model import create_model, DiffusionTransformer, ModelConfig
from src.core.diffusion import DiscreteDiffusion


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def small_config(temp_dir):
    """Create a small config for testing."""
    return DataConfig(
        subset_size=50,
        vocab_size=256,
        max_seq_len=32,
        data_dir=temp_dir,
        tokenizer_path=f"{temp_dir}/tokenizer.json",
        train_data_path=f"{temp_dir}/train_tokens.pt",
        val_data_path=f"{temp_dir}/val_tokens.pt",
        config_path=f"{temp_dir}/config.json",
    )


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset."""
    class MockDataset:
        def __init__(self, texts):
            self.data = [{"text": t} for t in texts]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return {"text": [d["text"] for d in self.data[idx]]}
            return self.data[idx]

        def __iter__(self):
            return iter(self.data)

        def select(self, indices):
            return MockDataset([self.data[i]["text"] for i in indices])

        def train_test_split(self, test_size, seed=None):
            n = len(self.data)
            split_idx = int(n * (1 - test_size))
            return {
                "train": MockDataset([d["text"] for d in self.data[:split_idx]]),
                "test": MockDataset([d["text"] for d in self.data[split_idx:]]),
            }

    texts = [
        "Once upon a time, there was a little cat named Whiskers.",
        "The cat loved to play in the sunny garden all day long.",
        "One day, Whiskers found a beautiful butterfly.",
        "The butterfly had colorful wings that sparkled.",
        "Whiskers tried to catch the butterfly but it flew away.",
        "The cat was sad but then found a ball of yarn.",
        "Playing with yarn made Whiskers very happy.",
        "At night, the cat curled up by the warm fire.",
        "Dreams of butterflies filled the cat's sleep.",
        "The next morning, Whiskers woke up ready for adventure.",
        "A bird sang outside the window.",
        "Whiskers watched with curious eyes.",
        "The garden was full of flowers.",
        "Bees buzzed from flower to flower.",
        "It was a perfect summer day.",
        "Whiskers stretched and yawned.",
        "Time for breakfast, thought the cat.",
        "A bowl of milk waited in the kitchen.",
        "Whiskers lapped it up happily.",
        "Then it was time to explore again.",
    ]
    return MockDataset(texts)


@pytest.fixture
def tiny_model():
    """Create a tiny model for integration tests."""
    return create_model("tiny", vocab_size=256, max_seq_len=32)


@pytest.fixture
def diffusion():
    """Create a diffusion process for integration tests."""
    return DiscreteDiffusion(vocab_size=256, mask_token_id=3, pad_token_id=0)


# =============================================================================
# Integration Tests: Data Pipeline
# =============================================================================

class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""

    def test_load_stories_with_mock(self, mock_hf_dataset, small_config):
        """Test load_stories with mocked HuggingFace dataset."""
        with patch('src.data.data_prep.load_dataset', return_value=mock_hf_dataset):
            train_ds, val_ds = load_stories(small_config)

            assert len(train_ds) > 0
            assert len(val_ds) > 0
            assert len(train_ds) + len(val_ds) <= len(mock_hf_dataset)

    def test_full_data_pipeline_with_mock(self, mock_hf_dataset, small_config):
        """Test the complete data pipeline with mocked dataset."""
        with patch('src.data.data_prep.load_dataset', return_value=mock_hf_dataset):
            # Run main pipeline
            train_tokens, val_tokens, tokenizer = data_prep_main(small_config)

            # Verify outputs
            assert train_tokens.shape[1] == small_config.max_seq_len
            assert val_tokens.shape[1] == small_config.max_seq_len

            # Verify files were created
            assert os.path.exists(small_config.tokenizer_path)
            assert os.path.exists(small_config.train_data_path)
            assert os.path.exists(small_config.val_data_path)
            assert os.path.exists(small_config.config_path)

            # Verify config file contents
            with open(small_config.config_path) as f:
                saved_config = json.load(f)
            assert saved_config["vocab_size"] == small_config.vocab_size
            assert "train_stats" in saved_config
            assert "val_stats" in saved_config

    def test_tokenizer_vocab_size_respected(self, mock_hf_dataset, small_config):
        """Test that tokenizer respects vocab_size setting."""
        with patch('src.data.data_prep.load_dataset', return_value=mock_hf_dataset):
            train_ds, _ = load_stories(small_config)
            tokenizer = train_tokenizer(train_ds, small_config)

            # Vocab size should not exceed configured size
            assert tokenizer.get_vocab_size() <= small_config.vocab_size

    def test_tokens_within_bounds(self, mock_hf_dataset, small_config):
        """Test that all tokens are within vocabulary bounds."""
        with patch('src.data.data_prep.load_dataset', return_value=mock_hf_dataset):
            train_tokens, val_tokens, tokenizer = data_prep_main(small_config)

            vocab_size = tokenizer.get_vocab_size()

            # All tokens should be valid
            assert (train_tokens >= 0).all()
            assert (train_tokens < vocab_size).all()
            assert (val_tokens >= 0).all()
            assert (val_tokens < vocab_size).all()


# =============================================================================
# Integration Tests: Model + Diffusion
# =============================================================================

class TestModelDiffusionIntegration:
    """Integration tests for model and diffusion working together."""

    def test_training_step(self, tiny_model, diffusion):
        """Test a complete training step."""
        # Create batch
        batch_size, seq_len = 4, 32
        tokens = torch.randint(5, 256, (batch_size, seq_len))

        # Training step
        tiny_model.train()
        loss, metrics = diffusion.training_losses(tiny_model, tokens)

        # Backward pass
        loss.backward()

        # Verify
        assert loss.item() > 0
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 < metrics["mask_rate"] < 1

        # Check gradients exist
        for param in tiny_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_training_multiple_steps(self, tiny_model, diffusion):
        """Test multiple training steps with optimizer."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            tokens = torch.randint(5, 256, (4, 32))

            optimizer.zero_grad()
            loss, _ = diffusion.training_losses(tiny_model, tokens)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should be computed successfully each time
        assert len(losses) == 5
        assert all(l > 0 for l in losses)

    def test_generation_after_training(self, tiny_model, diffusion):
        """Test that generation works after training."""
        # Quick training
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        for _ in range(3):
            tokens = torch.randint(5, 256, (4, 32))
            optimizer.zero_grad()
            loss, _ = diffusion.training_losses(tiny_model, tokens)
            loss.backward()
            optimizer.step()

        # Generate
        tiny_model.eval()
        samples = diffusion.sample(
            tiny_model,
            batch_size=2,
            seq_len=16,
            num_steps=10,
            device="cpu"
        )

        assert samples.shape == (2, 16)
        assert (samples >= 0).all()
        assert (samples < 256).all()

    def test_model_eval_vs_train_mode(self, tiny_model, diffusion):
        """Test that model behaves correctly in eval vs train mode."""
        tokens = torch.randint(5, 256, (2, 32))
        t = torch.rand(2)

        # Train mode
        tiny_model.train()
        out_train = tiny_model(tokens, t)

        # Eval mode
        tiny_model.eval()
        with torch.no_grad():
            out_eval = tiny_model(tokens, t)

        # Shapes should match
        assert out_train.shape == out_eval.shape

    def test_sample_with_trajectory(self, tiny_model, diffusion):
        """Test trajectory sampling for visualization."""
        samples, trajectory = diffusion.sample_with_trajectory(
            tiny_model,
            batch_size=2,
            seq_len=16,
            num_steps=10,
            device="cpu"
        )

        # Verify trajectory
        assert len(trajectory) == 11  # Initial + 10 steps
        assert trajectory[0].shape == samples.shape
        assert torch.equal(trajectory[-1], samples)

        # Masks should decrease
        initial_masks = (trajectory[0] == diffusion.mask_token_id).sum()
        final_masks = (trajectory[-1] == diffusion.mask_token_id).sum()
        assert final_masks < initial_masks


# =============================================================================
# Integration Tests: End-to-End Pipeline
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end tests for the complete training pipeline."""

    def test_data_to_training(self, mock_hf_dataset, temp_dir):
        """Test from data preparation to training step."""
        # Prepare data
        config = DataConfig(
            subset_size=20,
            vocab_size=256,
            max_seq_len=32,
            data_dir=temp_dir,
            tokenizer_path=f"{temp_dir}/tokenizer.json",
            train_data_path=f"{temp_dir}/train_tokens.pt",
            val_data_path=f"{temp_dir}/val_tokens.pt",
            config_path=f"{temp_dir}/config.json",
        )

        with patch('src.data.data_prep.load_dataset', return_value=mock_hf_dataset):
            train_tokens, val_tokens, tokenizer = data_prep_main(config)

        # Create model with matching vocab
        vocab_size = tokenizer.get_vocab_size()
        model = create_model("tiny", vocab_size=vocab_size, max_seq_len=32)
        diffusion = DiscreteDiffusion(
            vocab_size=vocab_size,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )

        # Training step
        batch = train_tokens[:4]
        loss, metrics = diffusion.training_losses(model, batch)

        assert loss.item() > 0
        assert metrics["mask_rate"] > 0

    def test_save_and_load_model(self, tiny_model, diffusion, temp_dir):
        """Test saving and loading model checkpoint."""
        # Train briefly
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        tokens = torch.randint(5, 256, (4, 32))

        loss, _ = diffusion.training_losses(tiny_model, tokens)
        loss.backward()
        optimizer.step()

        # Save
        checkpoint_path = f"{temp_dir}/model.pt"
        torch.save({
            "model_state_dict": tiny_model.state_dict(),
            "config": tiny_model.config,
        }, checkpoint_path)

        # Load into new model
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_model = DiffusionTransformer(checkpoint["config"])
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify same outputs
        tiny_model.eval()
        new_model.eval()
        t = torch.rand(4)

        with torch.no_grad():
            out1 = tiny_model(tokens, t)
            out2 = new_model(tokens, t)

        assert torch.allclose(out1, out2)


# =============================================================================
# Integration Tests: Different Configurations
# =============================================================================

class TestConfigurationVariations:
    """Tests for different model and diffusion configurations."""

    @pytest.mark.parametrize("config_name", ["tiny", "small"])
    def test_model_configs(self, config_name, diffusion):
        """Test different model configurations."""
        model = create_model(config_name, vocab_size=256, max_seq_len=32)
        tokens = torch.randint(5, 256, (2, 32))

        loss, metrics = diffusion.training_losses(model, tokens)

        assert loss.item() > 0

    @pytest.mark.parametrize("schedule", ["cosine", "linear"])
    def test_noise_schedules(self, tiny_model, schedule):
        """Test different noise schedules."""
        diffusion = DiscreteDiffusion(vocab_size=256, schedule=schedule)
        tokens = torch.randint(5, 256, (2, 32))

        loss, metrics = diffusion.training_losses(tiny_model, tokens)

        assert loss.item() > 0

    @pytest.mark.parametrize("num_steps", [5, 25, 50])
    def test_sampling_steps(self, tiny_model, diffusion, num_steps):
        """Test different numbers of sampling steps."""
        samples = diffusion.sample(
            tiny_model,
            batch_size=2,
            seq_len=16,
            num_steps=num_steps,
            device="cpu"
        )

        assert samples.shape == (2, 16)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
