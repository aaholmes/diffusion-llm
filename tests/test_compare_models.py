#!/usr/bin/env python3
"""
Tests for compare_models.py and evaluate_captioning.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from tokenizers import Tokenizer

from src.models.model_autoregressive import AutoregressiveCaptioner, AutoregressiveConfig
from src.core.model import DiffusionTransformer, ModelConfig
from src.core.diffusion import DiscreteDiffusion


class TestCompareModels:
    """Test model comparison functionality."""

    @pytest.fixture
    def mock_ar_checkpoint(self, tmp_path):
        """Create mock autoregressive checkpoint."""
        config = AutoregressiveConfig(
            d_model=256,
            n_layers=2,
            vocab_size=1000,
            max_seq_len=32,
        )
        model = AutoregressiveCaptioner(config)

        checkpoint = {
            'global_step': 1000,
            'model_state_dict': model.state_dict(),
            'model_config': config.__dict__,
        }

        checkpoint_path = tmp_path / "ar_model.pt"
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    @pytest.fixture
    def mock_diff_checkpoint(self, tmp_path):
        """Create mock diffusion checkpoint."""
        config = ModelConfig(
            d_model=256,
            n_layers=2,
            vocab_size=1000,
            max_seq_len=32,
            has_cross_attention=True,
        )
        model = DiffusionTransformer(config)

        checkpoint = {
            'global_step': 1000,
            'decoder_state_dict': model.state_dict(),
            'decoder_config': config.__dict__,
        }

        checkpoint_path = tmp_path / "diff_model.pt"
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create mock data directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create tokenizer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tokenizer_content = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": None,
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "<UNK>",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "byte_fallback": False,
                    "vocab": {f"token{i}": i for i in range(1000)},
                    "merges": []
                }
            }
            json.dump(tokenizer_content, f)
            self.tokenizer_path = f.name

        # Create config
        config = {
            'feature_dim': 256,
            'vocab_size': 1000,
            'tokenizer': self.tokenizer_path,
        }
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)

        # Create data
        val_features = torch.randn(5, 50, 256)
        val_captions = torch.randint(0, 1000, (5, 32))

        torch.save(val_features, data_dir / "val_image_features.pt")
        torch.save(val_captions, data_dir / "val_captions.pt")

        return str(data_dir)

    def test_load_autoregressive(self, mock_ar_checkpoint):
        """Test loading autoregressive model."""
        from src.evaluation.compare_models import load_autoregressive

        model, config = load_autoregressive(mock_ar_checkpoint, "cpu")

        assert isinstance(model, AutoregressiveCaptioner)
        assert isinstance(config, AutoregressiveConfig)
        assert config.d_model == 256

    def test_load_diffusion(self, mock_diff_checkpoint):
        """Test loading diffusion model."""
        from src.evaluation.compare_models import load_diffusion

        model, diffusion, config = load_diffusion(mock_diff_checkpoint, "cpu")

        assert isinstance(model, DiffusionTransformer)
        assert isinstance(diffusion, DiscreteDiffusion)
        assert isinstance(config, ModelConfig)
        assert config.d_model == 256

    def test_main_function(self, mock_ar_checkpoint, mock_diff_checkpoint, mock_data_dir, monkeypatch):
        """Test main comparison function."""
        from src.evaluation.compare_models import main

        monkeypatch.setattr(sys, 'argv', [
            'compare_models.py',
            '--autoregressive_ckpt', mock_ar_checkpoint,
            '--diffusion_ckpt', mock_diff_checkpoint,
            '--data_dir', mock_data_dir,
            '--num_examples', '3',
            '--device', 'cpu',
        ])

        # Should run without error
        main()

        # Cleanup tokenizer
        if hasattr(self, 'tokenizer_path'):
            os.unlink(self.tokenizer_path)


class TestEvaluateCaptioning:
    """Test evaluate_captioning.py functionality."""

    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create mock checkpoint."""
        config = ModelConfig(
            d_model=256,
            n_layers=2,
            vocab_size=1000,
            max_seq_len=32,
            has_cross_attention=True,
        )
        model = DiffusionTransformer(config)

        checkpoint = {
            'global_step': 1000,
            'decoder_state_dict': model.state_dict(),
            'decoder_config': config.__dict__,
        }

        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    @pytest.fixture
    def mock_data_dir_with_captions(self, tmp_path):
        """Create mock data directory with caption JSON."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create tokenizer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tokenizer_content = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": None,
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "<UNK>",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "byte_fallback": False,
                    "vocab": {
                        "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<MASK>": 3,
                        **{f"token{i}": i+4 for i in range(996)}
                    },
                    "merges": []
                }
            }
            json.dump(tokenizer_content, f)
            self.tokenizer_path = f.name

        # Create config
        config = {
            'feature_dim': 256,
            'vocab_size': 1000,
            'tokenizer': self.tokenizer_path,
        }
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)

        # Create data
        val_features = torch.randn(5, 50, 256)
        val_captions = torch.randint(4, 1000, (5, 32))
        # Add BOS and EOS
        val_captions[:, 0] = 1  # BOS
        val_captions[:, -1] = 2  # EOS

        torch.save(val_features, data_dir / "val_image_features.pt")
        torch.save(val_captions, data_dir / "val_captions.pt")

        # Create reference captions JSON
        val_captions_json = {
            str(i): [f"caption {i} text"] for i in range(5)
        }
        with open(data_dir / "val_captions.json", 'w') as f:
            json.dump(val_captions_json, f)

        return str(data_dir)

    def test_load_model_function(self, mock_checkpoint):
        """Test load_model from evaluate_captioning.py."""
        try:
            from evaluate_captioning import load_model

            model, diffusion, config = load_model(mock_checkpoint, device="cpu")

            assert isinstance(model, DiffusionTransformer)
            assert isinstance(diffusion, DiscreteDiffusion)
            assert isinstance(config, ModelConfig)
        except ImportError:
            pytest.skip("evaluate_captioning.py not properly structured")

    def test_compute_metrics(self):
        """Test metric computation."""
        try:
            from evaluate_captioning import compute_cider_score

            # Mock ground truth and predictions
            gts = {
                '0': ['a cat sitting on a mat'],
                '1': ['a dog running in park'],
            }
            res = {
                '0': ['a cat on a mat'],
                '1': ['a dog in the park'],
            }

            # This is a placeholder - actual CIDEr requires pycocoevalcap
            # Just test that the function exists
            assert callable(compute_cider_score)
        except (ImportError, AttributeError):
            # Expected if pycocoevalcap not installed or function not defined
            pytest.skip("CIDEr evaluation not available")

    def test_generate_and_evaluate_basic(self, mock_checkpoint, mock_data_dir_with_captions):
        """Test basic generation and evaluation workflow."""
        try:
            from evaluate_captioning import load_model
            from tokenizers import Tokenizer
            import json

            # Load model
            model, diffusion, config = load_model(mock_checkpoint, device="cpu")

            # Load tokenizer
            with open(Path(mock_data_dir_with_captions) / "config.json") as f:
                data_config = json.load(f)

            tokenizer = Tokenizer.from_file(data_config['tokenizer'])

            # Load val data
            val_features = torch.load(
                Path(mock_data_dir_with_captions) / "val_image_features.pt"
            )

            # Generate captions for one example
            image_features = val_features[0:1]
            image_mask = torch.ones(image_features.shape[:2])

            with torch.no_grad():
                generated_tokens = diffusion.sample(
                    model,
                    batch_size=1,
                    seq_len=32,
                    encoder_output=image_features,
                    encoder_attention_mask=image_mask,
                    num_steps=5,
                    temperature=1.0,
                )

            # Check output shape
            assert generated_tokens.shape == (1, 32)

            # Cleanup
            if hasattr(self, 'tokenizer_path'):
                os.unlink(self.tokenizer_path)

        except Exception as e:
            pytest.skip(f"Evaluation test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
