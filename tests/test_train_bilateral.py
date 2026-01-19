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
        assert config.batch_size == 32
        assert config.max_steps == 20000

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
