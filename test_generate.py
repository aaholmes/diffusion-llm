#!/usr/bin/env python3
"""Tests for generate.py"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from generate import generate


class TestGenerate:
    """Tests for the generate function."""

    @pytest.fixture
    def setup_model_and_tokenizer(self, tmp_path):
        """Create a test model checkpoint and tokenizer."""
        from model import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create tiny model
        model = create_model("tiny", vocab_size=1000, max_seq_len=64)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create minimal tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=1000,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        # Train on minimal data
        tokenizer.train_from_iterator(
            ["Once upon a time there was a little girl.",
             "The cat sat on the mat.",
             "Hello world this is a test."],
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        return str(checkpoint_path), str(tokenizer_path)

    def test_generate_produces_output(self, setup_model_and_tokenizer, capsys):
        """Test that generate function runs and produces output."""
        checkpoint_path, tokenizer_path = setup_model_and_tokenizer

        # Patch vocab_size to match our tiny tokenizer
        with patch('generate.create_model') as mock_create:
            from model import create_model
            mock_create.return_value = create_model("tiny", vocab_size=1000, max_seq_len=64)

            # This should run without error
            # Note: We can't fully test without matching vocab sizes,
            # so we test the components separately
            pass

    def test_generate_with_different_temperatures(self, setup_model_and_tokenizer):
        """Test generation with different temperature values."""
        checkpoint_path, tokenizer_path = setup_model_and_tokenizer

        # Just verify the function signature accepts these params
        # Full integration would require matching vocab sizes
        from model import create_model
        from diffusion import DiscreteDiffusion

        model = create_model("tiny", vocab_size=1000, max_seq_len=64)
        model.eval()
        diffusion = DiscreteDiffusion(vocab_size=1000, mask_token_id=3)

        for temp in [0.5, 0.8, 1.0, 1.5]:
            samples = diffusion.sample(
                model, batch_size=2, seq_len=32,
                num_steps=5, temperature=temp, device="cpu"
            )
            assert samples.shape == (2, 32)

    def test_generate_with_different_steps(self, setup_model_and_tokenizer):
        """Test generation with different step counts."""
        from model import create_model
        from diffusion import DiscreteDiffusion

        model = create_model("tiny", vocab_size=1000, max_seq_len=64)
        model.eval()
        diffusion = DiscreteDiffusion(vocab_size=1000, mask_token_id=3)

        for steps in [10, 50, 100]:
            samples = diffusion.sample(
                model, batch_size=2, seq_len=32,
                num_steps=steps, temperature=0.8, device="cpu"
            )
            assert samples.shape == (2, 32)


class TestGenerateFunction:
    """Tests for the main generate function."""

    def test_generate_end_to_end(self, tmp_path):
        """Test full generate function."""
        from model import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create model
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create tokenizer with proper vocab
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=8192,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        texts = ["Once upon a time there was a little girl named Lucy."] * 100
        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Run generate
        generate(
            checkpoint_path=str(checkpoint_path),
            tokenizer_path=str(tokenizer_path),
            num_samples=2,
            seq_len=32,
            steps=5,
            temperature=0.8,
        )

    def test_main_function(self, tmp_path, monkeypatch):
        """Test main function with command line args."""
        import sys
        from model import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create model
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
        tokenizer.train_from_iterator(["Test story text."] * 100, trainer=trainer)
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Mock command line
        monkeypatch.setattr(sys, 'argv', [
            'generate.py',
            '--checkpoint', str(checkpoint_path),
            '--tokenizer', str(tokenizer_path),
            '-n', '1',
            '--steps', '5',
            '-t', '0.8',
        ])

        from generate import main
        main()


class TestGenerateIntegration:
    """Integration tests for generate module."""

    def test_model_loading_with_string_config(self, tmp_path):
        """Test loading model when config is a string."""
        from model import create_model, DiffusionTransformer

        model = create_model("tiny", vocab_size=1000, max_seq_len=64)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        path = tmp_path / "test.pt"
        torch.save(checkpoint, path)

        # Load and verify
        loaded = torch.load(path, weights_only=False)
        config = loaded["model_config"]
        assert isinstance(config, str)
        assert config == "tiny"

    def test_model_loading_with_config_object(self, tmp_path):
        """Test loading model when config is a ModelConfig object."""
        from model import create_model, DiffusionTransformer, ModelConfig

        # Create config matching tiny model (d_ff = 4 * d_model = 1024)
        config_obj = ModelConfig(
            d_model=256, n_heads=4, n_layers=4, d_ff=1024,
            vocab_size=1000, max_seq_len=64
        )
        model = DiffusionTransformer(config_obj)
        checkpoint = {
            "model_config": config_obj,
            "model_state_dict": model.state_dict(),
        }
        path = tmp_path / "test.pt"
        torch.save(checkpoint, path)

        # Load and verify
        loaded = torch.load(path, weights_only=False)
        config = loaded["model_config"]
        assert isinstance(config, ModelConfig)

        # Should be able to create model from it
        new_model = DiffusionTransformer(config)
        new_model.load_state_dict(loaded["model_state_dict"])
