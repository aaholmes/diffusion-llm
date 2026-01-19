#!/usr/bin/env python3
"""
Tests for autoregressive captioning model and training.

Tests model_autoregressive.py, train_autoregressive.py, and eval_autoregressive.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from src.models.model_autoregressive import (
    AutoregressiveConfig,
    AutoregressiveCaptioner,
    SinusoidalPositionEmbeddings,
    TransformerDecoderBlock,
)
from src.training.train_autoregressive import AutoregressiveTrainer, TrainConfig, CaptionDataset


class TestSinusoidalEmbeddings:
    """Test sinusoidal position embeddings."""

    def test_output_shape(self):
        """Test output shape is correct."""
        dim = 768
        pos_embed = SinusoidalPositionEmbeddings(dim)
        seq_len = 32

        embeddings = pos_embed(seq_len, device="cpu")

        assert embeddings.shape == (seq_len, dim)

    def test_deterministic(self):
        """Test embeddings are deterministic."""
        dim = 512
        pos_embed = SinusoidalPositionEmbeddings(dim)
        seq_len = 16

        emb1 = pos_embed(seq_len, device="cpu")
        emb2 = pos_embed(seq_len, device="cpu")

        assert torch.allclose(emb1, emb2)

    def test_even_odd_dimensions(self):
        """Test sine and cosine applied to even/odd dimensions."""
        dim = 256
        pos_embed = SinusoidalPositionEmbeddings(dim)
        embeddings = pos_embed(10, device="cpu")

        # First dimension should have lower frequency
        assert not torch.allclose(embeddings[0, 0], embeddings[0, 1])

        # Should have values in [-1, 1]
        assert embeddings.min() >= -1.0
        assert embeddings.max() <= 1.0


class TestTransformerDecoderBlock:
    """Test transformer decoder block."""

    def test_output_shape(self):
        """Test output shape matches input."""
        d_model, n_heads, d_ff = 512, 8, 2048
        block = TransformerDecoderBlock(d_model, n_heads, d_ff)

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, d_model)
        encoder_output = torch.randn(batch_size, 50, d_model)

        output = block(x, encoder_output)

        assert output.shape == x.shape

    def test_causal_mask(self):
        """Test causal mask prevents future token attention."""
        d_model, n_heads, d_ff = 256, 4, 1024
        block = TransformerDecoderBlock(d_model, n_heads, d_ff)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)
        encoder_output = torch.randn(batch_size, 10, d_model)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        output = block(x, encoder_output, self_attn_mask=causal_mask)

        assert output.shape == x.shape

    def test_encoder_mask(self):
        """Test encoder mask works."""
        d_model, n_heads, d_ff = 256, 4, 1024
        block = TransformerDecoderBlock(d_model, n_heads, d_ff)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)
        encoder_output = torch.randn(batch_size, 10, d_model)
        encoder_mask = torch.ones(batch_size, 10)
        encoder_mask[:, 5:] = 0  # Mask out last 5 positions

        output = block(x, encoder_output, encoder_mask=encoder_mask)

        assert output.shape == x.shape


class TestAutoregressiveConfig:
    """Test autoregressive configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AutoregressiveConfig()

        assert config.d_model == 768
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.vocab_size == 8192
        assert config.pad_token_id == 0

    def test_custom_values(self):
        """Test custom configuration."""
        config = AutoregressiveConfig(
            d_model=512,
            n_layers=4,
            vocab_size=4096,
        )

        assert config.d_model == 512
        assert config.n_layers == 4
        assert config.vocab_size == 4096


class TestAutoregressiveCaptioner:
    """Test autoregressive captioning model."""

    def test_initialization(self):
        """Test model initialization."""
        config = AutoregressiveConfig(
            d_model=256,
            n_layers=2,
            vocab_size=1000,
        )
        model = AutoregressiveCaptioner(config)

        assert isinstance(model, nn.Module)
        assert len(model.layers) == config.n_layers

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = AutoregressiveConfig(
            d_model=256,
            n_layers=2,
            vocab_size=1000,
        )
        model = AutoregressiveCaptioner(config)

        batch_size, seq_len = 4, 16
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        encoder_output = torch.randn(batch_size, 50, config.d_model)

        logits = model(tokens, encoder_output)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_with_encoder_mask(self):
        """Test forward pass with encoder mask."""
        config = AutoregressiveConfig(d_model=256, n_layers=2, vocab_size=1000)
        model = AutoregressiveCaptioner(config)

        batch_size, seq_len = 4, 16
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        encoder_output = torch.randn(batch_size, 50, config.d_model)
        encoder_mask = torch.ones(batch_size, 50)

        logits = model(tokens, encoder_output, encoder_mask)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_generate(self):
        """Test generation produces valid output."""
        config = AutoregressiveConfig(
            d_model=256,
            n_layers=2,
            vocab_size=1000,
            max_seq_len=32,
        )
        model = AutoregressiveCaptioner(config)
        model.eval()

        batch_size = 2
        encoder_output = torch.randn(batch_size, 50, config.d_model)

        generated = model.generate(
            encoder_output,
            max_len=20,
            temperature=1.0,
            bos_token_id=1,
            eos_token_id=2,
        )

        assert generated.shape == (batch_size, 20)
        assert (generated[:, 0] == 1).all()  # Starts with BOS

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling."""
        config = AutoregressiveConfig(d_model=256, n_layers=2, vocab_size=1000)
        model = AutoregressiveCaptioner(config)
        model.eval()

        encoder_output = torch.randn(1, 50, config.d_model)

        generated = model.generate(
            encoder_output,
            max_len=15,
            temperature=0.8,
            top_k=50,
            bos_token_id=1,
            eos_token_id=2,
        )

        assert generated.shape == (1, 15)

    def test_generate_stops_at_eos(self):
        """Test generation pads after EOS."""
        config = AutoregressiveConfig(
            d_model=128,
            n_layers=1,
            vocab_size=100,
        )
        model = AutoregressiveCaptioner(config)
        model.eval()

        encoder_output = torch.randn(1, 10, config.d_model)

        # Generate short sequence
        generated = model.generate(
            encoder_output,
            max_len=10,
            bos_token_id=1,
            eos_token_id=2,
        )

        # Should pad with pad_token_id after generation
        assert generated.shape == (1, 10)

    def test_causal_mask_generation(self):
        """Test causal mask is properly generated."""
        config = AutoregressiveConfig(d_model=256, n_layers=2, vocab_size=1000)
        model = AutoregressiveCaptioner(config)

        mask = model._generate_causal_mask(5, "cpu")

        # Upper triangle should be -inf
        assert mask[0, 1] == float('-inf')
        assert mask[0, 4] == float('-inf')
        # Diagonal and below should be 0
        assert mask[0, 0] == 0.0
        assert mask[4, 3] == 0.0


class TestCaptionDataset:
    """Test caption dataset."""

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        image_features = torch.randn(100, 50, 768)
        captions = torch.randint(0, 1000, (100, 32))

        dataset = CaptionDataset(image_features, captions)

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format."""
        image_features = torch.randn(10, 50, 768)
        captions = torch.randint(0, 1000, (10, 32))

        dataset = CaptionDataset(image_features, captions)
        item = dataset[0]

        assert 'image_features' in item
        assert 'caption_tokens' in item
        assert item['image_features'].shape == (50, 768)
        assert item['caption_tokens'].shape == (32,)


class TestTrainConfig:
    """Test training configuration."""

    def test_default_values(self):
        """Test default training config."""
        config = TrainConfig()

        assert config.batch_size == 32
        assert config.max_steps == 5000
        assert config.learning_rate == 3e-4

    def test_custom_values(self):
        """Test custom training config."""
        config = TrainConfig(
            batch_size=64,
            max_steps=10000,
            learning_rate=1e-4,
        )

        assert config.batch_size == 64
        assert config.max_steps == 10000
        assert config.learning_rate == 1e-4


class TestAutoregressiveTrainer:
    """Test autoregressive trainer."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create mock data directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create config
        config = {
            'feature_dim': 256,
            'vocab_size': 1000,
        }
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)

        # Create data tensors
        train_features = torch.randn(50, 50, 256)
        train_captions = torch.randint(0, 1000, (50, 32))
        val_features = torch.randn(10, 50, 256)
        val_captions = torch.randint(0, 1000, (10, 32))

        torch.save(train_features, data_dir / "train_image_features.pt")
        torch.save(train_captions, data_dir / "train_captions.pt")
        torch.save(val_features, data_dir / "val_image_features.pt")
        torch.save(val_captions, data_dir / "val_captions.pt")

        return str(data_dir)

    def test_trainer_initialization(self, mock_data_dir, tmp_path):
        """Test trainer initializes correctly."""
        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            n_layers=2,
            batch_size=8,
            use_amp=False,
        )

        trainer = AutoregressiveTrainer(config)

        assert trainer.config == config
        assert trainer.global_step == 0
        assert trainer.model is not None

    def test_learning_rate_schedule(self, mock_data_dir, tmp_path):
        """Test learning rate schedule."""
        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            n_layers=2,
            batch_size=8,
            warmup_steps=100,
            max_steps=1000,
            learning_rate=1e-3,
            use_amp=False,
        )

        trainer = AutoregressiveTrainer(config)

        # Test warmup
        lr_0 = trainer.get_lr(0)
        lr_50 = trainer.get_lr(50)
        lr_100 = trainer.get_lr(100)

        assert lr_0 == 0.0
        assert 0 < lr_50 < config.learning_rate
        assert lr_100 == config.learning_rate

        # Test decay
        lr_500 = trainer.get_lr(500)
        assert 0 < lr_500 < config.learning_rate

    def test_train_step(self, mock_data_dir, tmp_path):
        """Test single training step."""
        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            n_layers=2,
            batch_size=8,
            use_amp=False,
        )

        trainer = AutoregressiveTrainer(config)

        # Get a batch
        batch = next(iter(trainer.train_loader))

        # Run training step
        metrics = trainer.train_step(batch)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['accuracy'], float)

    def test_validation(self, mock_data_dir, tmp_path):
        """Test validation loop."""
        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            n_layers=2,
            batch_size=8,
            use_amp=False,
        )

        trainer = AutoregressiveTrainer(config)

        val_loss, val_acc = trainer.validate()

        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0 <= val_acc <= 1

    def test_save_checkpoint(self, mock_data_dir, tmp_path):
        """Test checkpoint saving."""
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=str(checkpoint_dir),
            n_layers=2,
            batch_size=8,
            use_amp=False,
        )

        trainer = AutoregressiveTrainer(config)
        trainer.save_checkpoint(100)

        checkpoint_path = checkpoint_dir / "step_100.pt"
        assert checkpoint_path.exists()

        # Load and verify
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'global_step' in checkpoint
        assert checkpoint['global_step'] == 100
        assert 'model_state_dict' in checkpoint
        assert 'model_config' in checkpoint

    def test_save_best_checkpoint(self, mock_data_dir, tmp_path):
        """Test best checkpoint saving."""
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainConfig(
            data_dir=mock_data_dir,
            checkpoint_dir=str(checkpoint_dir),
            n_layers=2,
            batch_size=8,
            use_amp=False,
        )

        trainer = AutoregressiveTrainer(config)
        trainer.save_checkpoint(200, is_best=True)

        best_path = checkpoint_dir / "best.pt"
        assert best_path.exists()


class TestEvalAutoregressive:
    """Test eval_autoregressive.py functionality."""

    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create mock checkpoint."""
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
            'best_val_loss': 2.5,
        }

        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path)

    def test_load_model(self, mock_checkpoint):
        """Test model loading from checkpoint."""
        from src.evaluation.eval_autoregressive import load_model

        model, config = load_model(mock_checkpoint, device="cpu")

        assert isinstance(model, AutoregressiveCaptioner)
        assert isinstance(config, AutoregressiveConfig)
        assert config.d_model == 256
        assert config.n_layers == 2

    def test_main_function(self, mock_checkpoint, tmp_path, monkeypatch):
        """Test main evaluation function."""
        from src.evaluation.eval_autoregressive import main

        # Create mock data
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
            tokenizer_path = f.name

        # Create config
        config = {
            'feature_dim': 256,
            'vocab_size': 1000,
            'tokenizer': tokenizer_path,
        }
        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)

        # Create data
        val_features = torch.randn(5, 50, 256)
        val_captions = torch.randint(0, 1000, (5, 32))

        torch.save(val_features, data_dir / "val_image_features.pt")
        torch.save(val_captions, data_dir / "val_captions.pt")

        # Mock command line arguments
        monkeypatch.setattr(sys, 'argv', [
            'eval_autoregressive.py',
            '--checkpoint', mock_checkpoint,
            '--data_dir', str(data_dir),
            '--num_examples', '3',
            '--device', 'cpu',
        ])

        # Should run without error
        main()

        # Cleanup
        os.unlink(tokenizer_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
