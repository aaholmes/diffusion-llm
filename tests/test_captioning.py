#!/usr/bin/env python3
"""
Tests for image captioning components.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.core.model import DiffusionTransformer, ModelConfig, MODEL_CONFIGS
from src.core.diffusion import DiscreteDiffusion


class TestSyntheticDataGeneration:
    """Test synthetic caption data generation."""

    def test_generate_image(self):
        """Test synthetic image generation."""
        from src.data.prep_caption_synthetic import generate_image, COLOR_RGB

        # Test each shape
        for shape in ["square", "circle", "triangle", "rectangle"]:
            for size in ["small", "large"]:
                for color in COLOR_RGB.keys():
                    img = generate_image(color, shape, size)
                    assert img.size == (224, 224)
                    assert img.mode == 'RGB'

    def test_generate_caption(self):
        """Test caption generation."""
        from src.data.prep_caption_synthetic import generate_caption

        caption = generate_caption("red", "square", "small")
        assert isinstance(caption, str)
        assert "red" in caption.lower()
        assert "square" in caption.lower()
        assert len(caption) > 0

    def test_caption_templates(self):
        """Test different caption templates."""
        from src.data.prep_caption_synthetic import generate_caption

        # Generate multiple captions with same attributes
        captions = [generate_caption("blue", "circle", "large") for _ in range(10)]

        # Should get different templates
        assert len(set(captions)) > 1, "Should use multiple templates"

        # All should mention the attributes
        for caption in captions:
            assert "blue" in caption.lower()
            assert "circle" in caption.lower()


class TestCaptionTraining:
    """Test caption training components."""

    @pytest.fixture
    def tiny_caption_data(self, tmp_path):
        """Create tiny caption dataset."""
        # Create synthetic features and captions
        num_train = 50
        num_val = 10
        feature_dim = 768
        seq_len = 50
        max_caption_len = 32

        train_features = torch.randn(num_train, seq_len, feature_dim)
        train_captions = torch.randint(4, 100, (num_train, max_caption_len))
        train_captions[:, 0] = 1  # BOS
        train_captions[:, 10] = 2  # EOS
        train_captions[:, 11:] = 0  # PAD

        val_features = torch.randn(num_val, seq_len, feature_dim)
        val_captions = torch.randint(4, 100, (num_val, max_caption_len))
        val_captions[:, 0] = 1
        val_captions[:, 5] = 2
        val_captions[:, 6:] = 0

        # Save
        torch.save(train_features, tmp_path / "train_image_features.pt")
        torch.save(train_captions, tmp_path / "train_captions.pt")
        torch.save(val_features, tmp_path / "val_image_features.pt")
        torch.save(val_captions, tmp_path / "val_captions.pt")

        # Save config
        config = {
            "clip_model": "openai/clip-vit-base-patch32",
            "feature_dim": feature_dim,
            "seq_len": seq_len,
            "max_caption_len": max_caption_len,
            "vocab_size": 8192,
        }

        with open(tmp_path / "config.json", 'w') as f:
            json.dump(config, f)

        return tmp_path

    def test_caption_dataset(self, tiny_caption_data):
        """Test CaptionDataset."""
        from src.training.train_captioning import CaptionDataset

        features = torch.load(tiny_caption_data / "train_image_features.pt")
        captions = torch.load(tiny_caption_data / "train_captions.pt")

        dataset = CaptionDataset(features, captions)

        assert len(dataset) == len(captions)

        sample = dataset[0]
        assert 'image_features' in sample
        assert 'caption_tokens' in sample
        assert sample['image_features'].shape == features[0].shape
        assert sample['caption_tokens'].shape == captions[0].shape

    def test_caption_trainer_init(self, tiny_caption_data):
        """Test CaptionTrainer initialization."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        config = CaptionTrainConfig(
            data_dir=str(tiny_caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=8,
            max_steps=10,
            use_amp=False,
        )

        trainer = CaptionTrainer(config)

        assert trainer.decoder is not None
        assert trainer.diffusion is not None
        assert trainer.optimizer is not None
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')

    def test_caption_train_step(self, tiny_caption_data):
        """Test single training step."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        config = CaptionTrainConfig(
            data_dir=str(tiny_caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=8,
            max_steps=10,
            use_amp=False,
        )

        trainer = CaptionTrainer(config)

        # Get a batch
        batch = next(iter(trainer.train_loader))

        # Training step
        metrics = trainer.train_step(batch)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['loss'] > 0
        assert 0 <= metrics['accuracy'] <= 1

    def test_caption_validation(self, tiny_caption_data):
        """Test validation."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        config = CaptionTrainConfig(
            data_dir=str(tiny_caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=8,
            max_steps=10,
            use_amp=False,
        )

        trainer = CaptionTrainer(config)

        val_loss, val_acc = trainer.validate()

        assert val_loss > 0
        assert 0 <= val_acc <= 1

    def test_caption_save_checkpoint(self, tiny_caption_data, tmp_path):
        """Test checkpoint saving."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        checkpoint_dir = tmp_path / "checkpoints"

        config = CaptionTrainConfig(
            data_dir=str(tiny_caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=8,
            max_steps=10,
            checkpoint_dir=str(checkpoint_dir),
            use_amp=False,
        )

        trainer = CaptionTrainer(config)
        trainer.global_step = 100

        trainer.save_checkpoint(100, is_best=True)

        # Check files exist
        assert (checkpoint_dir / "step_100.pt").exists()
        assert (checkpoint_dir / "best.pt").exists()

        # Check checkpoint content
        checkpoint = torch.load(checkpoint_dir / "best.pt", weights_only=False)
        assert checkpoint['global_step'] == 100
        assert 'decoder_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint

    def test_caption_learning_rate_schedule(self, tiny_caption_data):
        """Test learning rate warmup schedule."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        config = CaptionTrainConfig(
            data_dir=str(tiny_caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            learning_rate=1e-3,
            warmup_steps=100,
            use_amp=False,
        )

        trainer = CaptionTrainer(config)

        # Test warmup
        lr_0 = trainer.get_lr(0)
        lr_50 = trainer.get_lr(50)
        lr_100 = trainer.get_lr(100)
        lr_200 = trainer.get_lr(200)

        assert lr_0 == 0.0
        assert 0 < lr_50 < config.learning_rate
        assert lr_100 == config.learning_rate
        # After warmup, cosine decay has begun
        assert 0 < lr_200 < config.learning_rate


class TestCaptionGeneration:
    """Test caption generation."""

    @pytest.fixture
    def tiny_caption_model(self, tmp_path):
        """Create tiny caption model and checkpoint."""
        feature_dim = 768

        # Create decoder
        config = ModelConfig(
            d_model=feature_dim,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=8192,
            max_seq_len=32,
            has_cross_attention=True,
        )

        decoder = DiffusionTransformer(config)

        # Create checkpoint
        checkpoint = {
            'decoder_state_dict': decoder.state_dict(),
            'decoder_config': config.__dict__,
            'train_config': {
                'max_caption_len': 32,
            }
        }

        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path, feature_dim

    def test_load_captioning_model(self, tiny_caption_model):
        """Test loading captioning model."""
        from src.generation.generate_caption import load_captioning_model

        checkpoint_path, _ = tiny_caption_model

        decoder, train_config = load_captioning_model(str(checkpoint_path))

        assert decoder is not None
        assert isinstance(train_config, dict)

    def test_generate_caption_function(self, tiny_caption_model):
        """Test caption generation function."""
        from src.generation.generate_caption import load_captioning_model, generate_caption

        checkpoint_path, feature_dim = tiny_caption_model

        decoder, train_config = load_captioning_model(str(checkpoint_path))

        # Create diffusion
        diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3, pad_token_id=0)

        # Create fake image features
        batch_size = 2
        seq_len = 50
        image_features = torch.randn(batch_size, seq_len, feature_dim)

        # Generate
        output = generate_caption(
            decoder, diffusion, image_features,
            max_len=32, steps=5,  # Use few steps for testing
            temperature=0.8,
            device="cpu",
        )

        assert output.shape == (batch_size, 32)
        assert output.dtype == torch.long


class TestCaptionIntegration:
    """Integration tests for full caption pipeline."""

    def test_end_to_end_synthetic(self, tmp_path):
        """Test full pipeline with synthetic data (minimal)."""
        from src.data.prep_caption_synthetic import generate_image, generate_caption as gen_cap_text
        import random
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        # Generate minimal synthetic dataset
        num_examples = 20
        feature_dim = 768
        seq_len = 50
        max_caption_len = 32

        # Mock CLIP features (would normally come from vision model)
        features = torch.randn(num_examples, seq_len, feature_dim)

        # Generate simple captions
        import os
        from tokenizers import Tokenizer
        if not os.path.exists("data_full/tokenizer.json"):
            pytest.skip("data_full/tokenizer.json not available")
        tokenizer = Tokenizer.from_file("data_full/tokenizer.json")
        pad_id, bos_id, eos_id = 0, 1, 2

        caption_tokens = []
        for _ in range(num_examples):
            text = "A red square"  # Simple caption
            encoded = tokenizer.encode(text)
            tokens = [bos_id] + encoded.ids + [eos_id]
            tokens = tokens[:max_caption_len]
            tokens = tokens + [pad_id] * (max_caption_len - len(tokens))
            caption_tokens.append(tokens)

        caption_tokens = torch.tensor(caption_tokens)

        # Save data
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        torch.save(features[:15], data_dir / "train_image_features.pt")
        torch.save(caption_tokens[:15], data_dir / "train_captions.pt")
        torch.save(features[15:], data_dir / "val_image_features.pt")
        torch.save(caption_tokens[15:], data_dir / "val_captions.pt")

        config = {
            "feature_dim": feature_dim,
            "seq_len": seq_len,
            "max_caption_len": max_caption_len,
            "vocab_size": 8192,
        }

        with open(data_dir / "config.json", 'w') as f:
            json.dump(config, f)

        # Train for a few steps
        train_config = CaptionTrainConfig(
            data_dir=str(data_dir),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=4,
            max_steps=5,
            eval_every=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            use_amp=False,
        )

        trainer = CaptionTrainer(train_config)
        trainer.train()

        # Check that checkpoint was saved
        assert (tmp_path / "checkpoints" / "final.pt").exists()


class TestExtractImageFeatures:
    """Test image feature extraction."""

    def test_extract_image_features_rgb(self, tmp_path):
        """Test feature extraction from RGB image."""
        from src.generation.generate_caption import extract_image_features
        from unittest.mock import MagicMock, patch

        # Create test image
        img = Image.new('RGB', (224, 224), color='red')

        # Mock vision model and processor
        mock_vision_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 50, 768)
        mock_vision_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}

        features = extract_image_features(img, mock_vision_model, mock_processor, device="cpu")

        assert features.shape == (1, 50, 768)

    def test_extract_image_features_rgba_conversion(self, tmp_path):
        """Test that RGBA images are converted to RGB."""
        from src.generation.generate_caption import extract_image_features
        from unittest.mock import MagicMock

        # Create RGBA image
        img = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))

        mock_vision_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 50, 768)
        mock_vision_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}

        # Should not raise an error (converts RGBA to RGB internally)
        features = extract_image_features(img, mock_vision_model, mock_processor, device="cpu")
        assert features.shape == (1, 50, 768)


class TestGenerateCaptionEdgeCases:
    """Test edge cases in caption generation."""

    @pytest.fixture
    def caption_model(self, tmp_path):
        """Create a caption model for testing."""
        feature_dim = 768

        config = ModelConfig(
            d_model=feature_dim,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=8192,
            max_seq_len=32,
            has_cross_attention=True,
        )

        decoder = DiffusionTransformer(config)

        checkpoint = {
            'decoder_state_dict': decoder.state_dict(),
            'decoder_config': config.__dict__,
            'train_config': {
                'max_caption_len': 32,
            }
        }

        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path, feature_dim, decoder

    def test_generate_caption_single_step(self, caption_model):
        """Test generation with only one step."""
        from src.generation.generate_caption import generate_caption

        _, feature_dim, decoder = caption_model
        diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3, pad_token_id=0)

        image_features = torch.randn(1, 50, feature_dim)

        output = generate_caption(
            decoder, diffusion, image_features,
            max_len=16, steps=1,  # Single step
            temperature=1.0,
            device="cpu",
        )

        assert output.shape == (1, 16)

    def test_generate_caption_high_temperature(self, caption_model):
        """Test generation with high temperature."""
        from src.generation.generate_caption import generate_caption

        _, feature_dim, decoder = caption_model
        diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3, pad_token_id=0)

        image_features = torch.randn(1, 50, feature_dim)

        output = generate_caption(
            decoder, diffusion, image_features,
            max_len=16, steps=3,
            temperature=2.0,  # High temperature
            device="cpu",
        )

        assert output.shape == (1, 16)

    def test_generate_caption_low_temperature(self, caption_model):
        """Test generation with low temperature."""
        from src.generation.generate_caption import generate_caption

        _, feature_dim, decoder = caption_model
        diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3, pad_token_id=0)

        image_features = torch.randn(1, 50, feature_dim)

        output = generate_caption(
            decoder, diffusion, image_features,
            max_len=16, steps=3,
            temperature=0.1,  # Low temperature
            device="cpu",
        )

        assert output.shape == (1, 16)

    def test_generate_caption_batch(self, caption_model):
        """Test generation with batch of images."""
        from src.generation.generate_caption import generate_caption

        _, feature_dim, decoder = caption_model
        diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3, pad_token_id=0)

        # Batch of 5 images
        image_features = torch.randn(5, 50, feature_dim)

        output = generate_caption(
            decoder, diffusion, image_features,
            max_len=20, steps=3,
            temperature=0.8,
            device="cpu",
        )

        assert output.shape == (5, 20)


class TestSyntheticDataEdgeCases:
    """Test edge cases in synthetic data generation."""

    def test_all_colors(self):
        """Test all colors are valid."""
        from src.data.prep_caption_synthetic import COLOR_RGB, COLORS

        for color in COLORS:
            assert color in COLOR_RGB, f"Color {color} missing from COLOR_RGB"
            rgb = COLOR_RGB[color]
            assert len(rgb) == 3
            assert all(0 <= c <= 255 for c in rgb)

    def test_all_shapes(self):
        """Test all shapes can be generated."""
        from src.data.prep_caption_synthetic import generate_image, SHAPES, COLORS, SIZES

        for shape in SHAPES:
            img = generate_image(COLORS[0], shape, SIZES[0])
            assert img is not None
            assert img.size == (224, 224)

    def test_all_sizes(self):
        """Test both sizes produce different images."""
        from src.data.prep_caption_synthetic import generate_image
        import numpy as np

        img_small = generate_image("red", "square", "small")
        img_large = generate_image("red", "square", "large")

        # Images should be different
        arr_small = np.array(img_small)
        arr_large = np.array(img_large)

        assert not np.array_equal(arr_small, arr_large), "Small and large should produce different images"

    def test_caption_contains_all_attributes(self):
        """Test captions contain color, shape, and optionally size."""
        from src.data.prep_caption_synthetic import generate_caption, COLORS, SHAPES, SIZES

        for color in COLORS:
            for shape in SHAPES:
                for size in SIZES:
                    caption = generate_caption(color, shape, size)
                    assert color in caption.lower()
                    assert shape in caption.lower()


class TestCaptionTrainingExtended:
    """Extended tests for caption training."""

    @pytest.fixture
    def caption_data(self, tmp_path):
        """Create caption dataset for testing."""
        num_train = 30
        num_val = 10
        feature_dim = 768
        seq_len = 50
        max_caption_len = 32

        train_features = torch.randn(num_train, seq_len, feature_dim)
        train_captions = torch.randint(4, 100, (num_train, max_caption_len))
        train_captions[:, 0] = 1
        train_captions[:, 10] = 2
        train_captions[:, 11:] = 0

        val_features = torch.randn(num_val, seq_len, feature_dim)
        val_captions = torch.randint(4, 100, (num_val, max_caption_len))
        val_captions[:, 0] = 1
        val_captions[:, 5] = 2
        val_captions[:, 6:] = 0

        torch.save(train_features, tmp_path / "train_image_features.pt")
        torch.save(train_captions, tmp_path / "train_captions.pt")
        torch.save(val_features, tmp_path / "val_image_features.pt")
        torch.save(val_captions, tmp_path / "val_captions.pt")

        config = {
            "feature_dim": feature_dim,
            "seq_len": seq_len,
            "max_caption_len": max_caption_len,
            "vocab_size": 8192,
        }

        with open(tmp_path / "config.json", 'w') as f:
            json.dump(config, f)

        return tmp_path

    def test_train_multiple_steps(self, caption_data, tmp_path):
        """Test training for multiple steps."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        config = CaptionTrainConfig(
            data_dir=str(caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=4,
            max_steps=3,
            eval_every=2,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            use_amp=False,
        )

        trainer = CaptionTrainer(config)
        trainer.train()

        # Should have trained
        assert trainer.global_step == 3

    def test_checkpoint_loading(self, caption_data, tmp_path):
        """Test checkpoint saving and loading."""
        from src.training.train_captioning import CaptionTrainer, CaptionTrainConfig

        checkpoint_dir = tmp_path / "checkpoints"

        config = CaptionTrainConfig(
            data_dir=str(caption_data),
            decoder_config="tiny",
            decoder_n_layers=2,
            batch_size=4,
            max_steps=2,
            checkpoint_dir=str(checkpoint_dir),
            use_amp=False,
        )

        # Train and save
        trainer1 = CaptionTrainer(config)
        trainer1.train()

        # Load checkpoint
        checkpoint_path = checkpoint_dir / "final.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'global_step' in checkpoint
        assert 'decoder_state_dict' in checkpoint
        assert 'train_config' in checkpoint


@pytest.mark.skipif(
    not os.path.exists("data_full/tokenizer.json"),
    reason="data_full/tokenizer.json not available"
)
class TestGenerateCaptionMain:
    """Test the generate_caption.py main function."""

    @pytest.fixture
    def validation_data(self, tmp_path):
        """Create validation data for testing."""
        feature_dim = 768
        seq_len = 50
        max_caption_len = 32
        num_val = 5

        val_features = torch.randn(num_val, seq_len, feature_dim)
        val_captions = torch.randint(4, 100, (num_val, max_caption_len))
        val_captions[:, 0] = 1  # BOS
        val_captions[:, 5] = 2  # EOS
        val_captions[:, 6:] = 0  # PAD

        torch.save(val_features, tmp_path / "val_image_features.pt")
        torch.save(val_captions, tmp_path / "val_captions.pt")

        return tmp_path

    @pytest.fixture
    def caption_checkpoint(self, tmp_path):
        """Create checkpoint for testing."""
        feature_dim = 768

        config = ModelConfig(
            d_model=feature_dim,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=8192,
            max_seq_len=32,
            has_cross_attention=True,
        )

        decoder = DiffusionTransformer(config)

        checkpoint = {
            'decoder_state_dict': decoder.state_dict(),
            'decoder_config': config.__dict__,
            'train_config': {
                'max_caption_len': 32,
            }
        }

        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def test_main_with_val_set(self, validation_data, caption_checkpoint, tmp_path, monkeypatch, capsys):
        """Test main function with validation set."""
        import sys
        from src.generation.generate_caption import main

        # Set up argv
        monkeypatch.setattr(sys, 'argv', [
            'generate_caption.py',
            '--checkpoint', str(caption_checkpoint),
            '--use_val_set',
            '--num_samples', '2',
            '--data_dir', str(validation_data),
            '--steps', '3',
        ])

        main()

        captured = capsys.readouterr()
        assert "Loading checkpoint" in captured.out
        assert "Ground truth" in captured.out
        assert "Generated" in captured.out

    def test_main_no_image_or_val_set(self, caption_checkpoint, monkeypatch, capsys):
        """Test main function error when neither image nor val_set provided."""
        import sys
        from src.generation.generate_caption import main

        monkeypatch.setattr(sys, 'argv', [
            'generate_caption.py',
            '--checkpoint', str(caption_checkpoint),
        ])

        main()

        captured = capsys.readouterr()
        assert "Must specify --image or --use_val_set" in captured.out

    def test_main_with_image(self, caption_checkpoint, tmp_path, monkeypatch, capsys):
        """Test main function with image file."""
        import sys
        from unittest.mock import MagicMock, patch
        from src.generation.generate_caption import main

        # Create test image
        test_img = Image.new('RGB', (224, 224), color='red')
        img_path = tmp_path / "test.jpg"
        test_img.save(img_path)

        # Mock CLIP model
        mock_vision_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 50, 768)
        mock_vision_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}

        with patch('src.generation.generate_caption.CLIPVisionModel') as mock_clip:
            with patch('src.generation.generate_caption.CLIPImageProcessor') as mock_clip_proc:
                mock_clip.from_pretrained.return_value = mock_vision_model
                mock_clip_proc.from_pretrained.return_value = mock_processor

                monkeypatch.setattr(sys, 'argv', [
                    'generate_caption.py',
                    '--checkpoint', str(caption_checkpoint),
                    '--image', str(img_path),
                    '--num_samples', '2',
                    '--steps', '3',
                ])

                main()

                captured = capsys.readouterr()
                assert "Generated Captions:" in captured.out


@pytest.mark.skipif(
    not os.path.exists("data_full/tokenizer.json"),
    reason="data_full/tokenizer.json not available"
)
class TestPrepSyntheticMain:
    """Test prep_caption_synthetic.py main function."""

    def test_main_generates_data(self, tmp_path, monkeypatch):
        """Test main function generates synthetic data."""
        import sys
        from unittest.mock import MagicMock, patch

        # Mock CLIP model
        mock_vision_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 50, 768)
        mock_vision_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}

        with patch('src.data.prep_caption_synthetic.CLIPVisionModel') as mock_clip_model:
            with patch('src.data.prep_caption_synthetic.CLIPImageProcessor') as mock_clip_processor:
                mock_clip_model.from_pretrained.return_value = mock_vision_model
                mock_clip_processor.from_pretrained.return_value = mock_processor

                monkeypatch.setattr(sys, 'argv', [
                    'prep_caption_synthetic.py',
                    '--output_dir', str(tmp_path / "output"),
                    '--num_train', '10',
                    '--num_val', '5',
                    '--max_caption_len', '32',
                ])

                from src.data.prep_caption_synthetic import main
                main()

                # Check files were created
                output_dir = tmp_path / "output"
                assert (output_dir / "train_image_features.pt").exists()
                assert (output_dir / "train_captions.pt").exists()
                assert (output_dir / "val_image_features.pt").exists()
                assert (output_dir / "val_captions.pt").exists()
                assert (output_dir / "config.json").exists()

                # Check sizes
                train_features = torch.load(output_dir / "train_image_features.pt")
                train_captions = torch.load(output_dir / "train_captions.pt")
                assert train_features.shape[0] == 10
                assert train_captions.shape[0] == 10


class TestModelConfigWithCrossAttention:
    """Test model configuration with cross-attention."""

    def test_cross_attention_config(self):
        """Test ModelConfig with cross-attention enabled."""
        config = ModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=1000,
            max_seq_len=64,
            has_cross_attention=True,
        )

        model = DiffusionTransformer(config)

        # Check that cross-attention layers exist
        assert model.config.has_cross_attention

    def test_model_forward_with_encoder_output(self):
        """Test model forward pass with encoder output."""
        config = ModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=1000,
            max_seq_len=64,
            has_cross_attention=True,
        )

        model = DiffusionTransformer(config)

        batch_size = 2
        seq_len = 16
        encoder_seq_len = 50

        x = torch.randint(0, 1000, (batch_size, seq_len))
        t = torch.rand(batch_size)
        encoder_output = torch.randn(batch_size, encoder_seq_len, 256)

        output = model(x, t, encoder_output=encoder_output)

        assert output.shape == (batch_size, seq_len, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
