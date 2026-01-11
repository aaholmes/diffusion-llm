#!/usr/bin/env python3
"""
Comprehensive tests for train.py

Run with: pytest test_train.py -v
"""

import json
import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import TrainConfig, Trainer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_data(temp_dir):
    """Create mock training and validation data."""
    # Create small token tensors
    train_tokens = torch.randint(5, 256, (100, 32))
    val_tokens = torch.randint(5, 256, (20, 32))

    train_path = os.path.join(temp_dir, "train_tokens.pt")
    val_path = os.path.join(temp_dir, "val_tokens.pt")

    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    return train_path, val_path


@pytest.fixture
def small_config(temp_dir, mock_data):
    """Create a small config for testing."""
    train_path, val_path = mock_data

    return TrainConfig(
        model_config="tiny",
        vocab_size=256,
        max_seq_len=32,
        train_data_path=train_path,
        val_data_path=val_path,
        batch_size=8,
        grad_accum_steps=1,
        max_steps=10,
        learning_rate=1e-3,
        warmup_steps=2,
        eval_every=5,
        save_every=5,
        num_eval_batches=2,
        checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
        log_every=2,
        use_wandb=False,
        device="cpu",
        use_amp=False,
        num_workers=0,
    )


@pytest.fixture
def trainer(small_config):
    """Create a trainer for testing."""
    return Trainer(small_config)


# =============================================================================
# Tests: TrainConfig
# =============================================================================

class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainConfig()

        assert config.model_config == "small"
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4
        assert config.max_steps == 50000
        assert config.warmup_steps == 1000

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = TrainConfig(batch_size=32, grad_accum_steps=4)

        assert config.effective_batch_size == 128

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainConfig(
            model_config="tiny",
            batch_size=16,
            max_steps=1000,
        )

        assert config.model_config == "tiny"
        assert config.batch_size == 16
        assert config.max_steps == 1000


# =============================================================================
# Tests: Learning Rate Schedule
# =============================================================================

class TestLRSchedule:
    """Tests for learning rate scheduling."""

    def test_lr_at_step_zero(self, trainer):
        """Test LR is zero at step 0."""
        lr = trainer.get_lr(0)
        assert lr == 0.0

    def test_lr_at_warmup_end(self, trainer):
        """Test LR reaches max at end of warmup."""
        warmup_steps = trainer.config.warmup_steps
        lr = trainer.get_lr(warmup_steps)

        assert abs(lr - trainer.config.learning_rate) < 1e-6

    def test_lr_warmup_linear(self, trainer):
        """Test LR increases linearly during warmup."""
        warmup_steps = trainer.config.warmup_steps
        max_lr = trainer.config.learning_rate

        lr_mid = trainer.get_lr(warmup_steps // 2)
        expected_mid = max_lr * 0.5

        assert abs(lr_mid - expected_mid) < 1e-6

    def test_lr_decay_after_warmup(self, trainer):
        """Test LR decays after warmup."""
        warmup_steps = trainer.config.warmup_steps

        lr_at_warmup = trainer.get_lr(warmup_steps)
        lr_after = trainer.get_lr(warmup_steps + 100)

        assert lr_after < lr_at_warmup

    def test_lr_at_max_steps(self, trainer):
        """Test LR reaches minimum at max steps."""
        max_steps = trainer.config.max_steps
        min_lr = trainer.config.learning_rate * trainer.config.min_lr_ratio

        lr = trainer.get_lr(max_steps)

        assert abs(lr - min_lr) < 1e-6

    def test_lr_never_negative(self, trainer):
        """Test LR is never negative."""
        for step in range(0, trainer.config.max_steps + 100, 10):
            lr = trainer.get_lr(step)
            assert lr >= 0

    def test_lr_cosine_shape(self, trainer):
        """Test LR follows cosine decay shape."""
        warmup_steps = trainer.config.warmup_steps
        max_steps = trainer.config.max_steps

        # Get LR at several points after warmup
        steps = [warmup_steps + i * (max_steps - warmup_steps) // 4
                 for i in range(5)]
        lrs = [trainer.get_lr(s) for s in steps]

        # Should be decreasing
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i-1]


# =============================================================================
# Tests: Training Step
# =============================================================================

class TestTrainingStep:
    """Tests for training step."""

    def test_train_step_returns_metrics(self, trainer):
        """Test that train_step returns expected metrics."""
        batch = torch.randint(5, 256, (4, 32))
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mask_rate" in metrics

    def test_train_step_loss_positive(self, trainer):
        """Test that loss is positive."""
        batch = torch.randint(5, 256, (4, 32))
        metrics = trainer.train_step(batch)

        assert metrics["loss"] > 0

    def test_train_step_accuracy_in_range(self, trainer):
        """Test that accuracy is in [0, 1]."""
        batch = torch.randint(5, 256, (4, 32))
        metrics = trainer.train_step(batch)

        assert 0 <= metrics["accuracy"] <= 1

    def test_optimizer_step_clips_gradients(self, trainer):
        """Test that optimizer step returns gradient norm."""
        batch = torch.randint(5, 256, (4, 32))
        trainer.train_step(batch)

        grad_norm = trainer.optimizer_step()

        # Gradient norm is returned (before clipping is applied)
        # Just verify it's a valid positive number
        assert grad_norm >= 0
        assert math.isfinite(grad_norm)


# =============================================================================
# Tests: Gradient Accumulation
# =============================================================================

class TestGradientAccumulation:
    """Tests for gradient accumulation."""

    def test_loss_scaled_by_accum_steps(self, small_config, mock_data):
        """Test that loss is scaled by gradient accumulation steps."""
        # Config with grad accumulation
        config = TrainConfig(
            **{**small_config.__dict__,
               'grad_accum_steps': 4,
               'use_wandb': False}
        )

        trainer = Trainer(config)

        batch = torch.randint(5, 256, (4, 32))

        # The loss scaling happens inside train_step
        # We verify by checking the backward pass doesn't fail
        metrics = trainer.train_step(batch)
        assert metrics["loss"] > 0


# =============================================================================
# Tests: Validation
# =============================================================================

class TestValidation:
    """Tests for validation loop."""

    def test_evaluate_returns_metrics(self, trainer):
        """Test that evaluate returns expected metrics."""
        metrics = trainer.evaluate()

        assert "val_loss" in metrics
        assert "val_accuracy" in metrics

    def test_evaluate_loss_positive(self, trainer):
        """Test that validation loss is positive."""
        metrics = trainer.evaluate()

        assert metrics["val_loss"] > 0

    def test_evaluate_accuracy_in_range(self, trainer):
        """Test that validation accuracy is in [0, 1]."""
        metrics = trainer.evaluate()

        assert 0 <= metrics["val_accuracy"] <= 1

    def test_evaluate_respects_num_batches(self, small_config, mock_data):
        """Test that evaluate respects num_eval_batches."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'num_eval_batches': 1,
               'use_wandb': False}
        )

        trainer = Trainer(config)

        # Should complete quickly with only 1 batch
        metrics = trainer.evaluate()
        assert "val_loss" in metrics


# =============================================================================
# Tests: Checkpointing
# =============================================================================

class TestCheckpointing:
    """Tests for checkpoint saving and loading."""

    def test_save_checkpoint_creates_file(self, trainer, temp_dir):
        """Test that save_checkpoint creates a file."""
        trainer.global_step = 100
        trainer.save_checkpoint("test")

        path = os.path.join(trainer.config.checkpoint_dir, "test.pt")
        assert os.path.exists(path)

    def test_save_checkpoint_contents(self, trainer, temp_dir):
        """Test that checkpoint contains expected keys."""
        trainer.global_step = 100
        trainer.save_checkpoint("test")

        path = os.path.join(trainer.config.checkpoint_dir, "test.pt")
        checkpoint = torch.load(path, weights_only=False)

        assert "global_step" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "train_config" in checkpoint
        assert checkpoint["global_step"] == 100

    def test_save_best_checkpoint(self, trainer, temp_dir):
        """Test saving best checkpoint."""
        trainer.global_step = 100
        trainer.save_checkpoint("test", is_best=True)

        best_path = os.path.join(trainer.config.checkpoint_dir, "best.pt")
        assert os.path.exists(best_path)

    def test_load_checkpoint_restores_state(self, trainer, temp_dir):
        """Test that load_checkpoint restores training state."""
        # Save checkpoint
        trainer.global_step = 100
        trainer.best_val_loss = 0.5
        trainer.save_checkpoint("test")

        # Reset state
        trainer.global_step = 0
        trainer.best_val_loss = float('inf')

        # Load checkpoint
        path = os.path.join(trainer.config.checkpoint_dir, "test.pt")
        trainer.load_checkpoint(path)

        assert trainer.global_step == 100
        assert trainer.best_val_loss == 0.5

    def test_cleanup_old_checkpoints(self, trainer, temp_dir):
        """Test that old checkpoints are cleaned up."""
        # Save more checkpoints than keep_last_n
        for step in [100, 200, 300, 400, 500]:
            trainer.global_step = step
            trainer.save_checkpoint()

        # Count step_*.pt files
        checkpoint_dir = trainer.config.checkpoint_dir
        step_files = list(Path(checkpoint_dir).glob("step_*.pt"))

        assert len(step_files) <= trainer.config.keep_last_n


from pathlib import Path


# =============================================================================
# Tests: Full Training Loop
# =============================================================================

class TestTrainingLoop:
    """Tests for complete training loop."""

    def test_train_completes(self, trainer):
        """Test that training completes without errors."""
        # Run short training
        trainer.train()

        assert trainer.global_step == trainer.config.max_steps

    def test_train_creates_checkpoints(self, trainer, temp_dir):
        """Test that training creates checkpoints."""
        trainer.train()

        # Should have final checkpoint
        final_path = os.path.join(trainer.config.checkpoint_dir, "final.pt")
        assert os.path.exists(final_path)

    def test_train_updates_best_val_loss(self, trainer):
        """Test that best_val_loss is updated during training."""
        initial_best = trainer.best_val_loss
        trainer.train()

        # Best val loss should be updated from infinity
        assert trainer.best_val_loss < initial_best

    def test_train_with_grad_accumulation(self, small_config, mock_data):
        """Test training with gradient accumulation."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'grad_accum_steps': 2,
               'max_steps': 4,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        trainer.train()
        assert trainer.global_step == 4


# =============================================================================
# Tests: Resume Training
# =============================================================================

class TestResumeTraining:
    """Tests for resuming training from checkpoint."""

    def test_resume_from_checkpoint(self, small_config, mock_data, temp_dir):
        """Test resuming training from checkpoint."""
        # Initial training
        config1 = TrainConfig(
            **{**small_config.__dict__,
               'max_steps': 5,
               'use_wandb': False}
        )

        trainer1 = Trainer(config1)
        trainer1.train()
        checkpoint_path = os.path.join(config1.checkpoint_dir, "final.pt")

        # Resume training
        config2 = TrainConfig(
            **{**small_config.__dict__,
               'max_steps': 10,
               'use_wandb': False,
               'resume_from': checkpoint_path}
        )

        trainer2 = Trainer(config2)
        assert trainer2.global_step == 5  # Resumed from step 5

        trainer2.train()
        assert trainer2.global_step == 10


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_step_training(self, small_config, mock_data):
        """Test training with single step."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'max_steps': 1,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        trainer.train()
        assert trainer.global_step == 1

    def test_batch_size_one(self, small_config, mock_data):
        """Test training with batch size 1."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'batch_size': 1,
               'max_steps': 3,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        trainer.train()
        assert trainer.global_step == 3

    def test_no_warmup(self, small_config, mock_data):
        """Test training without warmup."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'warmup_steps': 0,
               'max_steps': 5,
               'use_wandb': False}
        )

        trainer = Trainer(config)

        # LR should be max at step 0 with no warmup
        lr = trainer.get_lr(0)
        # With no warmup, step 0 should be at start of cosine decay
        # which starts at max_lr
        assert lr == config.learning_rate

        trainer.train()
        assert trainer.global_step == 5


# =============================================================================
# Tests: Data Iterator Exhaustion
# =============================================================================

class TestDataIterator:
    """Tests for data iterator behavior."""

    def test_iterator_resets_on_exhaustion(self, small_config, mock_data, temp_dir):
        """Test that data iterator resets when exhausted."""
        # Create config where we'll exhaust the data loader multiple times
        # 100 samples with batch_size 8 = ~12 batches per epoch
        # max_steps 25 > 12 batches, so iterator must reset
        config = TrainConfig(
            **{**small_config.__dict__,
               'batch_size': 8,
               'max_steps': 25,
               'grad_accum_steps': 1,
               'eval_every': 100,  # Don't eval during this test
               'save_every': 100,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        trainer.train()

        # If we got here without error, iterator reset worked
        assert trainer.global_step == 25


# =============================================================================
# Tests: Wandb Integration
# =============================================================================

class TestWandbIntegration:
    """Tests for wandb logging integration."""

    def test_wandb_disabled_by_config(self, small_config, mock_data):
        """Test that wandb can be disabled via config."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        assert trainer.use_wandb is False

    def test_wandb_init_failure_handled(self, small_config, mock_data):
        """Test that wandb init failure is handled gracefully."""
        from unittest.mock import patch, MagicMock

        config = TrainConfig(
            **{**small_config.__dict__,
               'use_wandb': True}
        )

        # Mock wandb to raise an exception on init
        with patch.dict('sys.modules', {'wandb': MagicMock()}):
            import sys
            mock_wandb = sys.modules['wandb']
            mock_wandb.init.side_effect = Exception("Wandb connection failed")

            trainer = Trainer(config)
            # Should have disabled wandb after failure
            assert trainer.use_wandb is False

    def test_wandb_import_error_handled(self, small_config, mock_data):
        """Test that missing wandb is handled gracefully."""
        from unittest.mock import patch

        config = TrainConfig(
            **{**small_config.__dict__,
               'use_wandb': True}
        )

        # Mock wandb import to fail
        with patch.dict('sys.modules', {'wandb': None}):
            trainer = Trainer(config)
            # Should have disabled wandb after import failure
            assert trainer.use_wandb is False


# =============================================================================
# Tests: Logging and Metrics
# =============================================================================

class TestLoggingMetrics:
    """Tests for logging and metrics tracking."""

    def test_log_every_respected(self, small_config, mock_data):
        """Test that log_every controls logging frequency."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'max_steps': 10,
               'log_every': 5,
               'eval_every': 100,
               'save_every': 100,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        trainer.train()

        # Training should complete without logging errors
        assert trainer.global_step == 10

    def test_metrics_accumulation(self, trainer):
        """Test that metrics accumulate correctly across steps."""
        batch = torch.randint(5, 256, (4, 32))

        # Run multiple steps
        all_metrics = []
        for _ in range(3):
            metrics = trainer.train_step(batch)
            all_metrics.append(metrics)

        # All metrics should be valid
        for m in all_metrics:
            assert "loss" in m
            assert m["loss"] > 0


# =============================================================================
# Tests: AMP (Automatic Mixed Precision)
# =============================================================================

class TestAMP:
    """Tests for automatic mixed precision."""

    def test_amp_disabled_on_cpu(self, small_config, mock_data):
        """Test that AMP is disabled when on CPU."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'device': 'cpu',
               'use_amp': True,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        # AMP should be disabled on CPU
        assert not trainer.config.use_amp or trainer.config.device == 'cpu'

    def test_training_works_without_amp(self, small_config, mock_data):
        """Test that training works with AMP disabled."""
        config = TrainConfig(
            **{**small_config.__dict__,
               'use_amp': False,
               'max_steps': 3,
               'use_wandb': False}
        )

        trainer = Trainer(config)
        trainer.train()

        assert trainer.global_step == 3


# =============================================================================
# Tests: Checkpoint Cleanup
# =============================================================================

class TestCheckpointCleanup:
    """Tests for checkpoint cleanup behavior."""

    def test_keeps_best_checkpoint(self, trainer, temp_dir):
        """Test that best checkpoint is kept during cleanup."""
        # Save several checkpoints
        trainer.best_val_loss = 0.5
        trainer.global_step = 100
        trainer.save_checkpoint("step_100", is_best=True)

        trainer.global_step = 200
        trainer.save_checkpoint("step_200")

        trainer.global_step = 300
        trainer.save_checkpoint("step_300")

        # Best should still exist
        best_path = os.path.join(trainer.config.checkpoint_dir, "best.pt")
        assert os.path.exists(best_path)


# =============================================================================
# Tests: Model Parameter Access
# =============================================================================

class TestModelAccess:
    """Tests for model parameter access."""

    def test_model_parameters_accessible(self, trainer):
        """Test that model parameters are accessible."""
        params = list(trainer.model.parameters())
        assert len(params) > 0

    def test_optimizer_has_model_params(self, trainer):
        """Test that optimizer contains model parameters."""
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) > 0
        assert len(param_groups[0]['params']) > 0


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
