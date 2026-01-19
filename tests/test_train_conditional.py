#!/usr/bin/env python3
"""Tests for train_conditional.py"""

import pytest
import torch
import json
from pathlib import Path
from dataclasses import asdict

from src.training.train_conditional import ConditionalTrainConfig, ConditionalTrainer


class TestConditionalTrainConfig:
    """Tests for ConditionalTrainConfig dataclass."""

    def test_default_values(self):
        config = ConditionalTrainConfig()
        assert config.denoiser_checkpoint == "checkpoints_long/final.pt"
        assert config.encoder_config == "small"
        assert config.encoder_n_layers == 4
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.vocab_size == 8192

    def test_custom_values(self):
        config = ConditionalTrainConfig(
            batch_size=32,
            max_steps=5000,
            learning_rate=5e-5,
        )
        assert config.batch_size == 32
        assert config.max_steps == 5000
        assert config.learning_rate == 5e-5

    def test_asdict(self):
        config = ConditionalTrainConfig()
        d = asdict(config)
        assert "denoiser_checkpoint" in d
        assert "encoder_n_layers" in d
        assert d["mask_token_id"] == 3


class TestConditionalTrainerSetup:
    """Tests for ConditionalTrainer initialization and setup."""

    @pytest.fixture
    def setup_training_environment(self, tmp_path):
        """Create pretrained model and conditional data for testing."""
        from src.core.model import create_model, ModelConfig

        # Create and save pretrained denoiser (without cross-attention)
        model = create_model("tiny", vocab_size=1000, max_seq_len=64)
        checkpoint = {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
        }
        denoiser_path = tmp_path / "denoiser.pt"
        torch.save(checkpoint, denoiser_path)

        # Create conditional data
        data_dir = tmp_path / "data_conditional"
        data_dir.mkdir()

        # Encoder inputs (first sentences)
        train_enc = torch.randint(1, 1000, (100, 32))
        val_enc = torch.randint(1, 1000, (20, 32))

        # Decoder targets (rest of story)
        train_dec = torch.randint(1, 1000, (100, 64))
        val_dec = torch.randint(1, 1000, (20, 64))

        torch.save(train_enc, data_dir / "train_encoder.pt")
        torch.save(val_enc, data_dir / "val_encoder.pt")
        torch.save(train_dec, data_dir / "train_decoder.pt")
        torch.save(val_dec, data_dir / "val_decoder.pt")

        # Config file
        config = {
            "num_train_pairs": 100,
            "num_val_pairs": 20,
            "max_encoder_len": 32,
            "max_decoder_len": 64,
        }
        with open(data_dir / "config.json", "w") as f:
            json.dump(config, f)

        return str(denoiser_path), str(data_dir), tmp_path

    def test_trainer_initialization(self, setup_training_environment):
        """Test that trainer initializes correctly."""
        denoiser_path, data_dir, tmp_path = setup_training_environment

        config = ConditionalTrainConfig(
            denoiser_checkpoint=denoiser_path,
            data_dir=data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=8,
            max_steps=10,
            vocab_size=1000,
            max_encoder_len=32,
            max_decoder_len=64,
            use_amp=False,
            num_workers=0,
        )

        trainer = ConditionalTrainer(config)

        assert trainer.encoder is not None
        assert trainer.decoder is not None
        assert trainer.decoder.config.has_cross_attention
        assert trainer.global_step == 0

    def test_decoder_has_cross_attention(self, setup_training_environment):
        """Test that decoder is created with cross-attention enabled."""
        denoiser_path, data_dir, tmp_path = setup_training_environment

        config = ConditionalTrainConfig(
            denoiser_checkpoint=denoiser_path,
            data_dir=data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=8,
            vocab_size=1000,
            max_encoder_len=32,
            max_decoder_len=64,
            use_amp=False,
            num_workers=0,
        )

        trainer = ConditionalTrainer(config)

        # Check decoder has cross-attention
        assert trainer.decoder.config.has_cross_attention
        for block in trainer.decoder.blocks:
            assert block.has_cross_attention
            assert hasattr(block, 'cross_attn')
            assert hasattr(block, 'norm_cross')

    def test_decoder_frozen_except_cross_attention(self, setup_training_environment):
        """Test that decoder is frozen except cross-attention layers."""
        denoiser_path, data_dir, tmp_path = setup_training_environment

        config = ConditionalTrainConfig(
            denoiser_checkpoint=denoiser_path,
            data_dir=data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=8,
            vocab_size=1000,
            max_encoder_len=32,
            max_decoder_len=64,
            use_amp=False,
            num_workers=0,
        )

        trainer = ConditionalTrainer(config)

        # Check embeddings are frozen
        assert not trainer.decoder.token_embedding.weight.requires_grad
        assert not trainer.decoder.position_embedding.weight.requires_grad

        # Check self-attention is frozen
        for block in trainer.decoder.blocks:
            for param in block.self_attn.parameters():
                assert not param.requires_grad
            for param in block.norm1.parameters():
                assert not param.requires_grad

        # Check cross-attention is trainable
        for block in trainer.decoder.blocks:
            for param in block.cross_attn.parameters():
                assert param.requires_grad
            for param in block.norm_cross.parameters():
                assert param.requires_grad

    def test_encoder_fully_trainable(self, setup_training_environment):
        """Test that encoder is fully trainable."""
        denoiser_path, data_dir, tmp_path = setup_training_environment

        config = ConditionalTrainConfig(
            denoiser_checkpoint=denoiser_path,
            data_dir=data_dir,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=8,
            vocab_size=1000,
            max_encoder_len=32,
            max_decoder_len=64,
            use_amp=False,
            num_workers=0,
        )

        trainer = ConditionalTrainer(config)

        # All encoder params should be trainable
        for param in trainer.encoder.parameters():
            assert param.requires_grad


class TestConditionalTrainerTraining:
    """Tests for training functionality."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create a trainer for testing."""
        from src.core.model import create_model

        # Create and save pretrained denoiser
        model = create_model("tiny", vocab_size=500, max_seq_len=32)
        checkpoint = {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
        }
        denoiser_path = tmp_path / "denoiser.pt"
        torch.save(checkpoint, denoiser_path)

        # Create conditional data
        data_dir = tmp_path / "data_conditional"
        data_dir.mkdir()

        torch.save(torch.randint(1, 500, (50, 16)), data_dir / "train_encoder.pt")
        torch.save(torch.randint(1, 500, (10, 16)), data_dir / "val_encoder.pt")
        torch.save(torch.randint(1, 500, (50, 32)), data_dir / "train_decoder.pt")
        torch.save(torch.randint(1, 500, (10, 32)), data_dir / "val_decoder.pt")

        config = ConditionalTrainConfig(
            denoiser_checkpoint=str(denoiser_path),
            data_dir=str(data_dir),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=4,
            max_steps=5,
            vocab_size=500,
            max_encoder_len=16,
            max_decoder_len=32,
            use_amp=False,
            num_workers=0,
            eval_every=2,
            save_every=5,
            log_every=1,
        )

        return ConditionalTrainer(config)

    def test_train_step(self, trainer):
        """Test single training step."""
        enc_input = torch.randint(1, 500, (4, 16))
        dec_target = torch.randint(1, 500, (4, 32))

        metrics = trainer.train_step(enc_input, dec_target)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mask_rate" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["mask_rate"] <= 1

    def test_evaluate(self, trainer):
        """Test evaluation."""
        val_metrics = trainer.evaluate()

        assert "val_loss" in val_metrics
        assert "val_accuracy" in val_metrics
        assert val_metrics["val_loss"] > 0
        assert 0 <= val_metrics["val_accuracy"] <= 1

    def test_get_lr_warmup(self, trainer):
        """Test learning rate warmup."""
        # During warmup
        lr_0 = trainer.get_lr(0)
        lr_mid = trainer.get_lr(trainer.config.warmup_steps // 2)
        lr_end = trainer.get_lr(trainer.config.warmup_steps)

        assert lr_0 == 0
        assert lr_mid < lr_end
        assert lr_end == pytest.approx(trainer.config.learning_rate, rel=0.01)

    def test_get_lr_decay(self, trainer):
        """Test learning rate cosine decay."""
        # Override config for this test to have warmup < max_steps
        trainer.config.warmup_steps = 2
        trainer.config.max_steps = 100

        warmup = trainer.config.warmup_steps
        max_steps = trainer.config.max_steps

        lr_after_warmup = trainer.get_lr(warmup)
        lr_mid = trainer.get_lr((warmup + max_steps) // 2)
        lr_end = trainer.get_lr(max_steps)

        assert lr_after_warmup > lr_mid
        assert lr_mid > lr_end
        assert lr_end >= trainer.config.learning_rate * trainer.config.min_lr_ratio

    def test_optimizer_step(self, trainer):
        """Test optimizer step with gradient clipping."""
        enc_input = torch.randint(1, 500, (4, 16))
        dec_target = torch.randint(1, 500, (4, 32))

        # Do a forward/backward pass
        trainer.train_step(enc_input, dec_target)

        # Optimizer step
        grad_norm = trainer.optimizer_step()

        assert isinstance(grad_norm, float)
        assert grad_norm >= 0

    def test_save_and_load_checkpoint(self, trainer, tmp_path):
        """Test checkpoint saving."""
        # Train a bit
        enc_input = torch.randint(1, 500, (4, 16))
        dec_target = torch.randint(1, 500, (4, 32))
        trainer.train_step(enc_input, dec_target)
        trainer.optimizer_step()
        trainer.global_step = 10

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")

        checkpoint_path = Path(trainer.config.checkpoint_dir) / "test_checkpoint.pt"
        assert checkpoint_path.exists()

        # Load and verify
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["global_step"] == 10
        assert "encoder_state_dict" in checkpoint
        assert "decoder_state_dict" in checkpoint
        assert "encoder_config" in checkpoint
        assert "decoder_config" in checkpoint

    def test_save_best_checkpoint(self, trainer):
        """Test saving best checkpoint."""
        trainer.save_checkpoint("test", is_best=True)

        best_path = Path(trainer.config.checkpoint_dir) / "best.pt"
        assert best_path.exists()


class TestConditionalTrainerGeneration:
    """Tests for generation functionality."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create a trainer for testing."""
        from src.core.model import create_model

        model = create_model("tiny", vocab_size=500, max_seq_len=32)
        checkpoint = {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
        }
        denoiser_path = tmp_path / "denoiser.pt"
        torch.save(checkpoint, denoiser_path)

        data_dir = tmp_path / "data_conditional"
        data_dir.mkdir()

        torch.save(torch.randint(1, 500, (50, 16)), data_dir / "train_encoder.pt")
        torch.save(torch.randint(1, 500, (10, 16)), data_dir / "val_encoder.pt")
        torch.save(torch.randint(1, 500, (50, 32)), data_dir / "train_decoder.pt")
        torch.save(torch.randint(1, 500, (10, 32)), data_dir / "val_decoder.pt")

        config = ConditionalTrainConfig(
            denoiser_checkpoint=str(denoiser_path),
            data_dir=str(data_dir),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=4,
            vocab_size=500,
            max_encoder_len=16,
            max_decoder_len=32,
            use_amp=False,
            num_workers=0,
        )

        return ConditionalTrainer(config)

    def test_generate_sample(self, trainer):
        """Test sample generation."""
        prompt = torch.randint(1, 500, (16,))

        output = trainer.generate_sample(prompt, steps=5)

        assert output.shape == (1, trainer.config.max_decoder_len)
        assert output.dtype == torch.long

    def test_generate_sample_batch(self, trainer):
        """Test batch sample generation."""
        prompts = torch.randint(1, 500, (3, 16))

        output = trainer.generate_sample(prompts, steps=5)

        assert output.shape == (3, trainer.config.max_decoder_len)


class TestConditionalTrainerFullLoop:
    """Integration tests for full training loop."""

    def test_short_training_run(self, tmp_path):
        """Test a very short training run completes without error."""
        from src.core.model import create_model

        # Setup
        model = create_model("tiny", vocab_size=500, max_seq_len=32)
        checkpoint = {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
        }
        denoiser_path = tmp_path / "denoiser.pt"
        torch.save(checkpoint, denoiser_path)

        data_dir = tmp_path / "data_conditional"
        data_dir.mkdir()

        torch.save(torch.randint(1, 500, (20, 16)), data_dir / "train_encoder.pt")
        torch.save(torch.randint(1, 500, (5, 16)), data_dir / "val_encoder.pt")
        torch.save(torch.randint(1, 500, (20, 32)), data_dir / "train_decoder.pt")
        torch.save(torch.randint(1, 500, (5, 32)), data_dir / "val_decoder.pt")

        config = ConditionalTrainConfig(
            denoiser_checkpoint=str(denoiser_path),
            data_dir=str(data_dir),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=4,
            max_steps=3,
            vocab_size=500,
            max_encoder_len=16,
            max_decoder_len=32,
            use_amp=False,
            num_workers=0,
            eval_every=2,
            save_every=3,
            log_every=1,
            warmup_steps=1,
        )

        trainer = ConditionalTrainer(config)
        trainer.train()

        # Check final checkpoint exists
        final_path = Path(config.checkpoint_dir) / "final.pt"
        assert final_path.exists()

        # Check training progressed
        assert trainer.global_step == 3


class TestWeightLoading:
    """Tests for weight loading from pretrained model."""

    def test_compatible_weights_loaded(self, tmp_path):
        """Test that compatible weights are loaded from pretrained model."""
        from src.core.model import create_model

        # Create pretrained model
        pretrained = create_model("tiny", vocab_size=500, max_seq_len=32)

        # Set some weights to specific values
        with torch.no_grad():
            pretrained.token_embedding.weight.fill_(1.0)

        checkpoint = {
            "model_config": pretrained.config,
            "model_state_dict": pretrained.state_dict(),
        }
        denoiser_path = tmp_path / "denoiser.pt"
        torch.save(checkpoint, denoiser_path)

        # Create data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        torch.save(torch.randint(1, 500, (10, 16)), data_dir / "train_encoder.pt")
        torch.save(torch.randint(1, 500, (5, 16)), data_dir / "val_encoder.pt")
        torch.save(torch.randint(1, 500, (10, 32)), data_dir / "train_decoder.pt")
        torch.save(torch.randint(1, 500, (5, 32)), data_dir / "val_decoder.pt")

        config = ConditionalTrainConfig(
            denoiser_checkpoint=str(denoiser_path),
            data_dir=str(data_dir),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            batch_size=4,
            vocab_size=500,
            max_encoder_len=16,
            max_decoder_len=32,
            use_amp=False,
            num_workers=0,
        )

        trainer = ConditionalTrainer(config)

        # Check that token embedding was loaded
        assert torch.allclose(
            trainer.decoder.token_embedding.weight,
            torch.ones_like(trainer.decoder.token_embedding.weight)
        )
