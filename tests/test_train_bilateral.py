#!/usr/bin/env python3
"""
Tests for train_bilateral.py - Bilateral SDD training functionality.

Run with: pytest tests/test_train_bilateral.py -v
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from src.training.train_bilateral import TrainConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_data_dir(temp_dir):
    """Create mock data directory with all required files."""
    data_dir = temp_dir

    # Create config.json
    config_data = {
        "vocab_size": 500,
        "max_seq_len": 32,
    }
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config_data, f)

    # Create tokenizer mock file
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    Path(tokenizer_path).write_text("{}")

    # Create mock token tensors
    train_tokens = torch.randint(6, 500, (100, 32))
    val_tokens = torch.randint(6, 500, (20, 32))

    torch.save(train_tokens, os.path.join(data_dir, "train_tokens.pt"))
    torch.save(val_tokens, os.path.join(data_dir, "val_tokens.pt"))

    return data_dir


# =============================================================================
# Tests: TrainConfig
# =============================================================================

class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainConfig()

        assert config.model_version == "v2"
        assert config.embed_dim == 64
        assert config.d_model == 384
        assert config.n_layers == 6
        assert config.k_schedule == (1, 2, 4, 8)
        assert config.batch_size == 64  # Updated for fair comparison
        assert config.max_steps == 50000  # Updated for fair comparison

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainConfig(
            model_version="v1",
            d_model=256,
            n_layers=4,
            max_steps=1000,
        )

        assert config.model_version == "v1"
        assert config.d_model == 256
        assert config.n_layers == 4
        assert config.max_steps == 1000

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainConfig(max_steps=100)
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict["max_steps"] == 100

    def test_k_schedule_default(self):
        """Test default k schedule."""
        config = TrainConfig()
        assert len(config.k_schedule) == 4
        assert config.k_schedule[0] == 1
        assert config.k_schedule[-1] == 8

    def test_k_warmup_steps_default(self):
        """Test default k warmup steps."""
        config = TrainConfig()
        assert config.k_warmup_steps == 2500


# =============================================================================
# Tests: BilateralTrainer Initialization
# =============================================================================

class TestBilateralTrainerInit:
    """Tests for BilateralTrainer initialization."""

    def test_trainer_creates_v2_model(self, mock_data_dir, temp_dir):
        """Test trainer creates v2 (bilateral) model by default."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            model_version="v2",
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()

            trainer = BilateralTrainer(config)

            assert trainer.model is not None
            # v2 model has readout_query parameter
            assert hasattr(trainer.model, 'readout_query')

    def test_trainer_creates_v1_model(self, mock_data_dir, temp_dir):
        """Test trainer creates v1 (aggregation) model when specified."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            model_version="v1",
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()

            trainer = BilateralTrainer(config)

            assert trainer.model is not None
            # v1 model doesn't have readout_query
            assert not hasattr(trainer.model, 'readout_query')

    def test_trainer_creates_diffusion(self, mock_data_dir, temp_dir):
        """Test trainer creates diffusion process."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()

            trainer = BilateralTrainer(config)

            assert trainer.diffusion is not None

    def test_trainer_creates_optimizer(self, mock_data_dir, temp_dir):
        """Test trainer creates optimizer."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()

            trainer = BilateralTrainer(config)

            assert trainer.optimizer is not None


# =============================================================================
# Tests: Training Step
# =============================================================================

class TestBilateralTrainerTrainStep:
    """Tests for BilateralTrainer.train_step()."""

    @pytest.fixture
    def trainer(self, mock_data_dir, temp_dir):
        """Create a trainer instance."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4),
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            return BilateralTrainer(config)

    def test_train_step_returns_metrics(self, trainer):
        """Test that train_step returns metrics dictionary."""
        # Create a batch
        batch = torch.randint(6, 500, (trainer.config.batch_size, 32))

        metrics = trainer.train_step(batch)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics


# =============================================================================
# Tests: K-Curriculum
# =============================================================================

class TestKCurriculum:
    """Tests for k-curriculum functionality."""

    def test_get_current_k_at_start(self, mock_data_dir, temp_dir):
        """Test k value at start of training."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4, 8),
            k_warmup_steps=10,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            # At step 0, k should be first in schedule
            trainer.global_step = 0
            k_start = trainer.get_current_k()
            assert k_start == 2

    def test_get_current_k_increases(self, mock_data_dir, temp_dir):
        """Test that k increases with step count."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4, 8),
            k_warmup_steps=10,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            # At step 0, k should be 2
            trainer.global_step = 0
            k_start = trainer.get_current_k()

            # After many steps, k should be higher
            trainer.global_step = 100
            k_end = trainer.get_current_k()

            assert k_end >= k_start


# =============================================================================
# Tests: Checkpointing
# =============================================================================

class TestCheckpointing:
    """Tests for checkpoint saving and loading."""

    def test_save_checkpoint(self, mock_data_dir, temp_dir):
        """Test saving checkpoint."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            # Save checkpoint
            trainer.save_checkpoint()

            # Check file exists
            ckpt_path = os.path.join(config.checkpoint_dir, "step_0.pt")
            assert os.path.exists(ckpt_path)

    def test_checkpoint_contains_required_keys(self, mock_data_dir, temp_dir):
        """Test checkpoint contains all required keys."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)
            trainer.global_step = 10

            trainer.save_checkpoint()

            ckpt_path = os.path.join(config.checkpoint_dir, "step_10.pt")
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "global_step" in checkpoint
            assert checkpoint["global_step"] == 10

    def test_save_best_checkpoint(self, mock_data_dir, temp_dir):
        """Test saving best checkpoint."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)
            trainer.global_step = 10

            trainer.save_checkpoint(is_best=True)

            # Check best checkpoint was saved
            best_path = os.path.join(config.checkpoint_dir, "best.pt")
            assert os.path.exists(best_path)


# =============================================================================
# Tests: TextDataset
# =============================================================================

class TestTextDataset:
    """Tests for TextDataset class."""

    def test_dataset_length(self, mock_data_dir):
        """Test dataset reports correct length."""
        from src.training.train_bilateral import TextDataset

        tokens_path = os.path.join(mock_data_dir, "train_tokens.pt")
        dataset = TextDataset(tokens_path)

        assert len(dataset) == 100

    def test_dataset_getitem(self, mock_data_dir):
        """Test dataset getitem returns tensor."""
        from src.training.train_bilateral import TextDataset

        tokens_path = os.path.join(mock_data_dir, "train_tokens.pt")
        dataset = TextDataset(tokens_path)

        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (32,)

    def test_dataset_max_samples(self, mock_data_dir):
        """Test dataset respects max_samples."""
        from src.training.train_bilateral import TextDataset

        tokens_path = os.path.join(mock_data_dir, "train_tokens.pt")
        dataset = TextDataset(tokens_path, max_samples=10)

        assert len(dataset) == 10


# =============================================================================
# Tests: Learning Rate Schedule
# =============================================================================

class TestLearningRateSchedule:
    """Tests for learning rate schedule."""

    @pytest.fixture
    def trainer(self, mock_data_dir, temp_dir):
        """Create a trainer instance."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            learning_rate=1e-3,
            warmup_steps=10,
            max_steps=100,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            return BilateralTrainer(config)

    def test_lr_at_step_zero(self, trainer):
        """Test LR is zero at step 0."""
        trainer.global_step = 0
        lr = trainer.get_lr()
        assert lr == 0.0

    def test_lr_at_warmup_end(self, trainer):
        """Test LR reaches max at end of warmup."""
        trainer.global_step = trainer.config.warmup_steps
        lr = trainer.get_lr()
        assert abs(lr - trainer.config.learning_rate) < 1e-6

    def test_lr_warmup_linear(self, trainer):
        """Test LR increases linearly during warmup."""
        warmup = trainer.config.warmup_steps
        max_lr = trainer.config.learning_rate

        trainer.global_step = warmup // 2
        lr_mid = trainer.get_lr()
        expected = max_lr * 0.5

        assert abs(lr_mid - expected) < 1e-6

    def test_lr_decays_after_warmup(self, trainer):
        """Test LR decays after warmup."""
        trainer.global_step = trainer.config.warmup_steps
        lr_at_warmup = trainer.get_lr()

        trainer.global_step = trainer.config.warmup_steps + 50
        lr_after = trainer.get_lr()

        assert lr_after < lr_at_warmup

    def test_lr_never_negative(self, trainer):
        """Test LR is never negative."""
        for step in range(0, trainer.config.max_steps + 10, 5):
            trainer.global_step = step
            lr = trainer.get_lr()
            assert lr >= 0


# =============================================================================
# Tests: Validation
# =============================================================================

class TestValidation:
    """Tests for validation method."""

    def test_validate_returns_metrics(self, mock_data_dir, temp_dir):
        """Test that validate returns metrics dictionary."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            metrics = trainer.validate()

            assert isinstance(metrics, dict)
            assert "val_loss" in metrics
            assert "val_acc" in metrics

    def test_validate_loss_positive(self, mock_data_dir, temp_dir):
        """Test validation loss is positive."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            metrics = trainer.validate()

            assert metrics["val_loss"] > 0


# =============================================================================
# Tests: Sample Generation
# =============================================================================

class TestSampleGeneration:
    """Tests for sample generation."""

    def test_generate_samples_returns_list(self, mock_data_dir, temp_dir):
        """Test that generate_samples returns list of strings."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            # Mock tokenizer with decode method
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test text"
            mock_tok.from_file.return_value = mock_tokenizer

            trainer = BilateralTrainer(config)

            samples = trainer.generate_samples(num_samples=2, seq_len=16)

            assert isinstance(samples, list)
            assert len(samples) == 2
            assert all(isinstance(s, str) for s in samples)

    def test_generate_samples_different_lengths(self, mock_data_dir, temp_dir):
        """Test generating with different sequence lengths."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            trainer = BilateralTrainer(config)

            # Should work with different lengths
            samples_short = trainer.generate_samples(num_samples=1, seq_len=8)
            samples_long = trainer.generate_samples(num_samples=1, seq_len=24)

            assert len(samples_short) == 1
            assert len(samples_long) == 1


# =============================================================================
# Tests: Training Loop
# =============================================================================

class TestTrainingLoop:
    """Tests for full training loop."""

    def test_train_runs_without_error(self, mock_data_dir, temp_dir):
        """Test that training runs without error."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            batch_size=8,
            max_steps=3,
            warmup_steps=1,
            log_every=1,
            val_every=2,
            save_every=5,
            generate_every=100,  # Don't generate during test
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            trainer = BilateralTrainer(config)
            trainer.train()

            # train() iterates from 0 to max_steps-1, so final global_step is max_steps-1
            assert trainer.global_step == config.max_steps - 1

    def test_train_updates_k_curriculum(self, mock_data_dir, temp_dir):
        """Test that k curriculum updates during training."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4),
            k_warmup_steps=2,  # Very short for testing
            batch_size=8,
            max_steps=5,
            warmup_steps=1,
            log_every=100,
            val_every=100,
            save_every=100,
            generate_every=100,
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            trainer = BilateralTrainer(config)

            initial_k = trainer.current_k
            trainer.train()

            # K should have increased during training
            assert trainer.current_k >= initial_k

    def test_train_saves_best_model(self, mock_data_dir, temp_dir):
        """Test that training saves best model on validation improvement."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2,),
            batch_size=8,
            max_steps=4,
            warmup_steps=1,
            log_every=100,
            val_every=2,  # Validate frequently
            save_every=100,
            generate_every=100,
            use_amp=False,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            trainer = BilateralTrainer(config)
            trainer.train()

            # Best checkpoint should exist
            best_path = os.path.join(config.checkpoint_dir, "best.pt")
            assert os.path.exists(best_path)


# =============================================================================
# Tests: Main Function
# =============================================================================

class TestMainFunction:
    """Tests for main function."""

    def test_main_argument_parsing(self, mock_data_dir, temp_dir, monkeypatch):
        """Test main function argument parsing."""
        import sys

        test_args = [
            'train_bilateral.py',
            '--data_dir', mock_data_dir,
            '--checkpoint_dir', os.path.join(temp_dir, 'ckpts'),
            '--model_version', 'v2',
            '--batch_size', '8',
            '--max_steps', '2',
            '--learning_rate', '0.001',
            '--d_model', '32',
            '--n_layers', '1',
            '--n_heads', '2',
            '--embed_dim', '16',
            '--max_seq_len', '32',
            '--k_schedule', '2,4',
            '--log_every', '100',
            '--val_every', '100',
            '--save_every', '100',
            '--generate_every', '100',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            from src.training.train_bilateral import main
            main()

            # Verify checkpoint dir was created
            assert os.path.exists(os.path.join(temp_dir, 'ckpts'))

    def test_main_k_schedule_parsing(self, mock_data_dir, temp_dir, monkeypatch):
        """Test that k_schedule is parsed correctly from command line."""
        import sys

        test_args = [
            'train_bilateral.py',
            '--data_dir', mock_data_dir,
            '--checkpoint_dir', os.path.join(temp_dir, 'ckpts'),
            '--batch_size', '8',
            '--max_steps', '1',
            '--k_schedule', '1,2,4,8',
            '--d_model', '32',
            '--n_layers', '1',
            '--n_heads', '2',
            '--embed_dim', '16',
            '--max_seq_len', '32',
            '--log_every', '100',
            '--val_every', '100',
            '--save_every', '100',
            '--generate_every', '100',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            from src.training.train_bilateral import main
            main()

    def test_main_v1_model_version(self, mock_data_dir, temp_dir, monkeypatch):
        """Test main with v1 model version."""
        import sys

        test_args = [
            'train_bilateral.py',
            '--data_dir', mock_data_dir,
            '--checkpoint_dir', os.path.join(temp_dir, 'ckpts'),
            '--model_version', 'v1',
            '--batch_size', '8',
            '--max_steps', '2',
            '--d_model', '32',
            '--n_layers', '1',
            '--n_heads', '2',
            '--embed_dim', '16',
            '--max_seq_len', '32',
            '--k_schedule', '2',
            '--log_every', '100',
            '--val_every', '100',
            '--save_every', '100',
            '--generate_every', '100',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "test"
            mock_tok.from_file.return_value = mock_tokenizer

            from src.training.train_bilateral import main
            main()


# =============================================================================
# Tests: Update K
# =============================================================================

class TestUpdateK:
    """Tests for k update functionality."""

    def test_update_k_changes_model_config(self, mock_data_dir, temp_dir):
        """Test that update_k changes model and sdd config."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4, 8),
            k_warmup_steps=10,  # Use 10 for cleaner steps
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            # Initial k
            assert trainer.current_k == 2

            # Move to step that should trigger k increase to 4
            # k_idx = min(step // warmup, len(k_schedule) - 1)
            # At step 10: k_idx = min(10 // 10, 2) = 1, so k = k_schedule[1] = 4
            trainer.global_step = 10
            trainer.update_k_if_needed()

            # K should have increased to 4
            assert trainer.current_k == 4
            assert trainer.model.config.k == 4
            assert trainer.sdd_config.k == 4

    def test_k_maxes_out_at_schedule_end(self, mock_data_dir, temp_dir):
        """Test that k stops at final schedule value."""
        from src.training.train_bilateral import BilateralTrainer

        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            embed_dim=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
            k_schedule=(2, 4, 8),
            k_warmup_steps=5,
        )

        with patch('tokenizers.Tokenizer') as mock_tok:
            mock_tok.from_file.return_value = MagicMock()
            trainer = BilateralTrainer(config)

            # Very high step count
            trainer.global_step = 1000
            k = trainer.get_current_k()

            # Should be capped at max
            assert k == 8
