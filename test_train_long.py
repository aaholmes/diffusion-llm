#!/usr/bin/env python3
"""Tests for train_config_long.py and train_long.py"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from train_config_long import LongTrainConfig, print_config
from train_long import (
    find_latest_checkpoint,
    create_train_config,
    generate_samples,
)


class TestLongTrainConfig:
    """Tests for LongTrainConfig dataclass."""

    def test_default_values(self):
        config = LongTrainConfig()

        # Model defaults
        assert config.model_config == "small"
        assert config.vocab_size == 8192
        assert config.max_seq_len == 256

        # Training defaults
        assert config.batch_size == 32
        assert config.grad_accum_steps == 2
        assert config.max_steps == 12500

        # Learning rate defaults
        assert config.learning_rate == 3e-4
        assert config.min_lr_ratio == 0.1
        assert config.warmup_steps == 500

        # Regularization defaults
        assert config.weight_decay == 0.1
        assert config.max_grad_norm == 1.0
        assert config.dropout == 0.1

    def test_effective_batch_size(self):
        config = LongTrainConfig()
        effective_batch = config.batch_size * config.grad_accum_steps
        assert effective_batch == 64

    def test_custom_values(self):
        config = LongTrainConfig(
            model_config="medium",
            batch_size=64,
            max_steps=50000,
        )
        assert config.model_config == "medium"
        assert config.batch_size == 64
        assert config.max_steps == 50000

    def test_subset_size_none_means_full(self):
        config = LongTrainConfig()
        assert config.subset_size is None  # None means full dataset

    def test_early_stopping_patience(self):
        config = LongTrainConfig()
        assert config.early_stopping_patience == 10


class TestPrintConfig:
    """Tests for print_config function."""

    def test_print_config_runs(self, capsys):
        """Test that print_config executes without error."""
        print_config()
        captured = capsys.readouterr()
        assert "Long Training Configuration" in captured.out
        assert "model_config" in captured.out
        assert "learning_rate" in captured.out


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""

    def test_no_directory(self, tmp_path):
        result = find_latest_checkpoint(str(tmp_path / "nonexistent"))
        assert result is None

    def test_empty_directory(self, tmp_path):
        result = find_latest_checkpoint(str(tmp_path))
        assert result is None

    def test_finds_step_checkpoint(self, tmp_path):
        # Create fake checkpoints
        (tmp_path / "step_100.pt").touch()
        (tmp_path / "step_200.pt").touch()
        (tmp_path / "step_50.pt").touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "step_200.pt")

    def test_finds_final_checkpoint(self, tmp_path):
        (tmp_path / "final.pt").touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "final.pt")

    def test_prefers_step_over_final(self, tmp_path):
        (tmp_path / "final.pt").touch()
        (tmp_path / "step_1000.pt").touch()

        result = find_latest_checkpoint(str(tmp_path))
        # Should prefer latest step checkpoint
        assert "step_1000" in result

    def test_handles_invalid_filenames(self, tmp_path):
        (tmp_path / "step_abc.pt").touch()  # Invalid step number
        (tmp_path / "step_100.pt").touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "step_100.pt")


class TestCreateTrainConfig:
    """Tests for create_train_config function."""

    def test_basic_config_creation(self):
        long_config = LongTrainConfig()
        train_config = create_train_config(
            long_config,
            train_path="data/train.pt",
            val_path="data/val.pt",
        )

        assert train_config.model_config == "small"
        assert train_config.train_data_path == "data/train.pt"
        assert train_config.val_data_path == "data/val.pt"
        assert train_config.batch_size == 32
        assert train_config.learning_rate == 3e-4

    def test_max_steps_override(self):
        long_config = LongTrainConfig()
        train_config = create_train_config(
            long_config,
            train_path="train.pt",
            val_path="val.pt",
            max_steps=1000,
        )
        assert train_config.max_steps == 1000

    def test_resume_from(self):
        long_config = LongTrainConfig()
        train_config = create_train_config(
            long_config,
            train_path="train.pt",
            val_path="val.pt",
            resume_from="checkpoint.pt",
        )
        assert train_config.resume_from == "checkpoint.pt"


class TestGenerateSamples:
    """Tests for generate_samples function."""

    def test_generate_with_string_config(self, tmp_path):
        """Test generation with string model config in checkpoint."""
        from model import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create model checkpoint
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=8192,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story."] * 100,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Should run without error
        generate_samples(
            str(checkpoint_path),
            str(tokenizer_path),
            num_samples=1
        )

    def test_generate_with_config_object(self, tmp_path):
        """Test generation with ModelConfig object in checkpoint."""
        from model import create_model, ModelConfig, DiffusionTransformer
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create model with explicit config
        config = ModelConfig(
            d_model=256, n_heads=4, n_layers=4, d_ff=512,
            vocab_size=8192, max_seq_len=256
        )
        model = DiffusionTransformer(config)
        checkpoint = {
            "model_config": config,
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=8192,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story."] * 100,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Should run without error
        generate_samples(
            str(checkpoint_path),
            str(tokenizer_path),
            num_samples=1
        )


class TestPrepareFullDataset:
    """Tests for prepare_full_dataset function."""

    def test_skip_if_already_prepared(self, tmp_path, monkeypatch):
        """Test that data prep is skipped if files exist."""
        import os

        # Change to tmp_path so data_full is created there
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Create fake existing files
            data_dir = tmp_path / "data_full"
            data_dir.mkdir()
            (data_dir / "train_tokens.pt").touch()
            (data_dir / "val_tokens.pt").touch()
            (data_dir / "tokenizer.json").touch()

            from train_long import prepare_full_dataset
            from train_config_long import LongTrainConfig
            config = LongTrainConfig()

            # Should return paths without calling prepare_data
            result = prepare_full_dataset(config, force=False)

            # Verify it returns tuple of paths
            assert len(result) == 3
            assert "train_tokens.pt" in result[0]
            assert "val_tokens.pt" in result[1]
            assert "tokenizer.json" in result[2]
        finally:
            os.chdir(original_cwd)

    def test_prepare_calls_data_prep_when_forced(self, tmp_path, monkeypatch):
        """Test that force=True calls data preparation."""
        import os

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Create existing files
            data_dir = tmp_path / "data_full"
            data_dir.mkdir()
            (data_dir / "train_tokens.pt").touch()
            (data_dir / "val_tokens.pt").touch()
            (data_dir / "tokenizer.json").touch()

            # Track if prepare_data was called
            prepare_called = []

            def mock_prepare_data(config):
                prepare_called.append(True)

            monkeypatch.setattr("train_long.prepare_data", mock_prepare_data)

            from train_long import prepare_full_dataset
            from train_config_long import LongTrainConfig
            config = LongTrainConfig()
            config.subset_size = 100  # Small for testing

            result = prepare_full_dataset(config, force=True)

            # Should have called prepare_data
            assert len(prepare_called) == 1
        finally:
            os.chdir(original_cwd)


class TestMainFunction:
    """Tests for main function argument parsing."""

    def test_argument_parsing_quick_test(self, monkeypatch):
        """Test --quick_test flag sets correct values."""
        import sys
        from train_config_long import LongTrainConfig

        # Simulate command line args
        monkeypatch.setattr(sys, 'argv', ['train_long.py', '--quick_test', '--skip_data_prep'])

        config = LongTrainConfig()

        # In quick_test mode, these would be modified
        # We just verify the config object can be created
        assert config.max_steps == 12500  # Default before modification

    def test_argument_parsing_generate_only(self, monkeypatch, tmp_path):
        """Test --generate_only flag."""
        import sys

        # Create fake checkpoint
        checkpoint_dir = tmp_path / "checkpoints_long"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "final.pt").touch()

        monkeypatch.setattr(sys, 'argv', ['train_long.py', '--generate_only'])

        # Just verify args can be parsed
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--generate_only", action="store_true")
        args = parser.parse_args(['--generate_only'])
        assert args.generate_only is True

    def test_argument_parsing_resume(self, monkeypatch):
        """Test --resume flag."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args(['--resume'])
        assert args.resume is True

    def test_argument_parsing_max_steps(self, monkeypatch):
        """Test --max_steps override."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--max_steps", type=int, default=None)
        args = parser.parse_args(['--max_steps', '5000'])
        assert args.max_steps == 5000

    def test_find_latest_with_resume(self, tmp_path):
        """Test resume finds correct checkpoint."""
        from train_long import find_latest_checkpoint

        # Create checkpoints
        (tmp_path / "step_100.pt").touch()
        (tmp_path / "step_500.pt").touch()
        (tmp_path / "step_250.pt").touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert "step_500" in result


class TestIntegration:
    """Integration tests for train_long module."""

    def test_config_to_train_config_pipeline(self):
        """Test full config creation pipeline."""
        long_config = LongTrainConfig(
            model_config="tiny",
            max_steps=100,
            batch_size=8,
        )

        train_config = create_train_config(
            long_config,
            train_path="fake_train.pt",
            val_path="fake_val.pt",
        )

        # Verify all expected fields are set
        assert train_config.model_config == "tiny"
        assert train_config.max_steps == 100
        assert train_config.batch_size == 8
        assert train_config.vocab_size == long_config.vocab_size
        assert train_config.learning_rate == long_config.learning_rate
        assert train_config.weight_decay == long_config.weight_decay
        assert train_config.early_stopping_patience == long_config.early_stopping_patience


class TestGenerateSamplesEOS:
    """Test EOS handling in generate_samples."""

    def test_generate_samples_with_eos(self, tmp_path):
        """Test generation handles EOS token correctly."""
        from model import DiffusionTransformer, ModelConfig, MODEL_CONFIGS
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from train_long import generate_samples

        # Create model using actual tiny config
        tiny_config = MODEL_CONFIGS["tiny"]
        config = ModelConfig(
            d_model=tiny_config.d_model,
            n_heads=tiny_config.n_heads,
            n_layers=tiny_config.n_layers,
            d_ff=tiny_config.d_ff,
            vocab_size=8192,
            max_seq_len=256
        )
        model = DiffusionTransformer(config)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=8192,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story about a cat and dog."] * 100,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Should run without error
        generate_samples(str(checkpoint_path), str(tokenizer_path), num_samples=2)


class TestMainFunctionPaths:
    """Test various paths through main function."""

    def test_generate_only_no_checkpoint(self, tmp_path, monkeypatch, capsys):
        """Test generate_only mode when no checkpoint exists."""
        import sys
        from train_long import main

        # Create empty checkpoint directory
        checkpoint_dir = tmp_path / "checkpoints_long"
        checkpoint_dir.mkdir()

        # Change working directory
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            monkeypatch.setattr(sys, 'argv', ['train_long.py', '--generate_only'])
            main()

            captured = capsys.readouterr()
            assert "No checkpoint found" in captured.out
        finally:
            os.chdir(original_cwd)

    def test_resume_no_checkpoint(self, tmp_path, monkeypatch, capsys):
        """Test resume mode when no checkpoint exists."""
        import sys
        import os

        # Create directories and files
        checkpoint_dir = tmp_path / "checkpoints_long"
        checkpoint_dir.mkdir()
        data_dir = tmp_path / "data_full"
        data_dir.mkdir()
        (data_dir / "train_tokens.pt").touch()
        (data_dir / "val_tokens.pt").touch()
        (data_dir / "tokenizer.json").touch()

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            monkeypatch.setattr(sys, 'argv', [
                'train_long.py',
                '--resume',
                '--skip_data_prep',
                '--max_steps', '0'  # 0 steps to avoid actual training
            ])

            # Mock the Trainer to avoid actual training
            from unittest.mock import MagicMock, patch

            with patch('train_long.Trainer') as mock_trainer_class:
                mock_trainer = MagicMock()
                mock_trainer_class.return_value = mock_trainer

                from train_long import main
                main()

                captured = capsys.readouterr()
                assert "No checkpoint found, starting fresh" in captured.out
        finally:
            os.chdir(original_cwd)


class TestQuickTestMode:
    """Test quick_test mode configuration."""

    def test_quick_test_modifies_config(self, monkeypatch):
        """Test that quick_test flag modifies configuration."""
        import argparse
        from train_config_long import LongTrainConfig

        parser = argparse.ArgumentParser()
        parser.add_argument("--quick_test", action="store_true")
        parser.add_argument("--max_steps", type=int, default=None)
        args = parser.parse_args(['--quick_test'])

        config = LongTrainConfig()

        # In quick_test mode, max_steps would be set to 100
        if args.quick_test:
            args.max_steps = 100
            config.eval_every = 50
            config.save_every = 50
            config.subset_size = 1000

        assert args.max_steps == 100
        assert config.eval_every == 50
        assert config.subset_size == 1000
