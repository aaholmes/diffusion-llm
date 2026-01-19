#!/usr/bin/env python3
"""
Tests for jetson_webcam_demo.py

Covers:
- CaptionGenerator class (PyTorch and ONNX backends)
- Image feature extraction
- Caption generation pipeline
- Overlay drawing
- Model loading
- Main function integration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pytest

import numpy as np
import torch
from PIL import Image

# Import module under test
import src.deployment.jetson_webcam_demo as demo


class TestCaptionGenerator:
    """Test CaptionGenerator class."""

    @pytest.fixture
    def mock_decoder(self):
        """Mock PyTorch decoder."""
        decoder = Mock()
        decoder.vocab_size = 8192
        return decoder

    @pytest.fixture
    def mock_vision_model(self):
        """Mock CLIP vision model."""
        model = Mock()
        outputs = Mock()
        outputs.last_hidden_state = torch.randn(1, 50, 768)
        model.return_value = outputs
        return model

    @pytest.fixture
    def mock_image_processor(self):
        """Mock CLIP image processor."""
        processor = Mock()
        processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
        return processor

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer."""
        tokenizer = Mock()
        tokenizer.token_to_id.side_effect = lambda x: {
            "<PAD>": 0, "<MASK>": 3, "<BOS>": 1, "<EOS>": 2
        }.get(x, 5)
        tokenizer.decode.return_value = "a red circle"
        tokenizer.get_vocab_size.return_value = 8192
        return tokenizer

    def test_pytorch_backend_initialization(self, mock_decoder, mock_vision_model,
                                           mock_image_processor, mock_tokenizer):
        """Test CaptionGenerator initialization with PyTorch backend."""
        generator = demo.CaptionGenerator(
            decoder=mock_decoder,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_len=12,
            steps=10,
            temperature=0.8,
            backend="pytorch",
        )

        assert generator.backend == "pytorch"
        assert generator.max_len == 12
        assert generator.steps == 10
        assert generator.temperature == 0.8
        assert generator.vocab_size == 8192
        assert generator.pad_id == 0
        assert generator.mask_id == 3

    def test_onnx_backend_initialization(self, mock_vision_model,
                                        mock_image_processor, mock_tokenizer):
        """Test CaptionGenerator initialization with ONNX backend."""
        mock_session = Mock()

        generator = demo.CaptionGenerator(
            decoder=None,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_len=12,
            steps=10,
            temperature=0.8,
            backend="onnx",
            onnx_session=mock_session,
        )

        assert generator.backend == "onnx"
        assert generator.onnx_session == mock_session
        assert generator.vocab_size == 8192

    def test_extract_features(self, mock_decoder, mock_vision_model,
                              mock_image_processor, mock_tokenizer):
        """Test CLIP feature extraction."""
        generator = demo.CaptionGenerator(
            decoder=mock_decoder,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            backend="pytorch",
        )

        # Create test image
        image = Image.new('RGB', (224, 224), color='red')

        # Extract features
        features = generator.extract_features(image)

        # Verify
        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 50, 768)
        mock_image_processor.assert_called_once()
        mock_vision_model.assert_called_once()

    def test_extract_features_converts_grayscale(self, mock_decoder, mock_vision_model,
                                                 mock_image_processor, mock_tokenizer):
        """Test that grayscale images are converted to RGB."""
        generator = demo.CaptionGenerator(
            decoder=mock_decoder,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            backend="pytorch",
        )

        # Create grayscale image
        image = Image.new('L', (224, 224), color=128)

        # Extract features
        features = generator.extract_features(image)

        # Should not raise error
        assert isinstance(features, np.ndarray)

    def test_generate_pytorch(self, mock_decoder, mock_vision_model,
                              mock_image_processor, mock_tokenizer):
        """Test caption generation with PyTorch backend."""
        # Mock decoder forward pass
        mock_logits = torch.randn(1, 12, 8192)
        mock_decoder.return_value = mock_logits

        generator = demo.CaptionGenerator(
            decoder=mock_decoder,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_len=12,
            steps=5,  # Few steps for speed
            temperature=0.8,
            backend="pytorch",
        )

        # Generate
        image_features = torch.randn(1, 50, 768)
        caption = generator.generate_pytorch(image_features)

        # Verify
        assert isinstance(caption, str)
        assert caption == "a red circle"
        assert mock_decoder.call_count == 5  # steps=5
        mock_tokenizer.decode.assert_called_once()

    def test_generate_onnx(self, mock_vision_model, mock_image_processor, mock_tokenizer):
        """Test caption generation with ONNX backend."""
        # Mock ONNX session
        mock_session = Mock()
        mock_logits = np.random.randn(1, 12, 8192).astype(np.float32)
        mock_session.run.return_value = [mock_logits]

        generator = demo.CaptionGenerator(
            decoder=None,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_len=12,
            steps=5,
            temperature=0.8,
            backend="onnx",
            onnx_session=mock_session,
        )

        # Generate
        image_features = np.random.randn(1, 50, 768).astype(np.float32)
        caption = generator.generate_onnx(image_features)

        # Verify
        assert isinstance(caption, str)
        assert caption == "a red circle"
        assert mock_session.run.call_count == 5
        mock_tokenizer.decode.assert_called_once()

    @patch('src.deployment.jetson_webcam_demo.time')
    def test_generate_full_pipeline(self, mock_time, mock_decoder, mock_vision_model,
                                    mock_image_processor, mock_tokenizer):
        """Test full generation pipeline from frame to caption."""
        mock_time.perf_counter.side_effect = [0.0, 0.5, 0.5, 2.0]  # Feature and gen times

        mock_decoder.return_value = torch.randn(1, 12, 8192)

        generator = demo.CaptionGenerator(
            decoder=mock_decoder,
            vision_model=mock_vision_model,
            image_processor=mock_image_processor,
            tokenizer=mock_tokenizer,
            device="cpu",
            max_len=12,
            steps=2,
            backend="pytorch",
        )

        # Create test frame (BGR format)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Generate
        caption, inference_time = generator.generate(frame)

        # Verify
        assert isinstance(caption, str)
        assert isinstance(inference_time, float)
        assert inference_time == 2.0  # Total time from mock

    def test_softmax(self):
        """Test softmax helper function."""
        x = np.array([[[1.0, 2.0, 3.0]]])
        result = demo.CaptionGenerator._softmax(x)

        # Check properties
        assert result.shape == x.shape
        assert np.allclose(result.sum(axis=-1), 1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_multinomial_sample(self):
        """Test multinomial sampling helper."""
        probs = np.array([[[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]])
        result = demo.CaptionGenerator._multinomial_sample(probs)

        # Check shape
        assert result.shape == (1, 2)
        assert result.dtype == np.int64
        assert np.all(result >= 0)
        assert np.all(result < 3)


class TestModelLoading:
    """Test model loading functions."""

    def test_load_pytorch_model(self, tmp_path):
        """Test loading PyTorch checkpoint."""
        from src.core.model import ModelConfig, DiffusionTransformer

        # Create mock checkpoint with actual model state
        config = ModelConfig(
            vocab_size=8192,
            max_seq_len=128,
            d_model=768,
            n_layers=6,
            n_heads=6,
        )

        # Create a real model to get proper state dict
        model = DiffusionTransformer(config)
        state_dict = model.state_dict()

        checkpoint = {
            'decoder_config': config,
            'decoder_state_dict': state_dict,
            'train_config': {'max_caption_len': 12},
        }

        checkpoint_path = tmp_path / "test.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load
        decoder, train_config = demo.load_pytorch_model(str(checkpoint_path), "cpu")

        # Verify
        assert decoder is not None
        assert train_config['max_caption_len'] == 12

    def test_load_onnx_model(self, tmp_path):
        """Test loading ONNX model."""
        # Create mock config
        config = {
            'd_model': 768,
            'n_layers': 6,
            'n_heads': 6,
            'vocab_size': 8192,
        }

        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Create dummy ONNX file
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b'dummy onnx')

        # Mock onnxruntime at import level
        import sys
        mock_ort = MagicMock()
        mock_session = Mock()
        mock_session.get_providers.return_value = ['CPUExecutionProvider']
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict(sys.modules, {'onnxruntime': mock_ort}):
            # Load
            session, loaded_config = demo.load_onnx_model(str(onnx_path), str(config_path))

            # Verify
            assert session == mock_session
            assert loaded_config == config


class TestDrawOverlay:
    """Test drawing caption overlay on frames."""

    def test_draw_caption_overlay_basic(self):
        """Test basic overlay drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        caption = "a red circle"
        inference_time = 1.5

        result = demo.draw_caption_overlay(frame, caption, inference_time)

        # Verify
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
        # Frame should have been modified (overlay added)
        assert not np.array_equal(result, frame)

    def test_draw_caption_overlay_long_text(self):
        """Test overlay with long caption (word wrapping)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        caption = "a very long caption that should wrap to multiple lines when displayed"
        inference_time = 2.0

        result = demo.draw_caption_overlay(frame, caption, inference_time)

        # Verify
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_draw_caption_overlay_empty_caption(self):
        """Test overlay with empty caption."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        caption = ""
        inference_time = 0.0

        result = demo.draw_caption_overlay(frame, caption, inference_time)

        # Should not crash
        assert result.shape == frame.shape


class TestMainFunction:
    """Test main function and CLI integration."""

    @patch('src.deployment.jetson_webcam_demo.cv2')
    @patch('src.deployment.jetson_webcam_demo.CLIPVisionModel')
    @patch('src.deployment.jetson_webcam_demo.CLIPImageProcessor')
    @patch('src.deployment.jetson_webcam_demo.Tokenizer')
    @patch('src.deployment.jetson_webcam_demo.load_pytorch_model')
    def test_main_pytorch_backend(self, mock_load, mock_tokenizer_cls,
                                  mock_processor_cls, mock_vision_cls, mock_cv2):
        """Test main function with PyTorch backend."""
        # Mock command line args
        with patch('sys.argv', ['prog', '--checkpoint', 'test.pt',
                               '--tokenizer', 'tok.json']):
            # Mock components
            mock_tokenizer = Mock()
            mock_tokenizer.token_to_id.side_effect = lambda x: {
                "<PAD>": 0, "<MASK>": 3, "<BOS>": 1, "<EOS>": 2
            }.get(x, 5)
            mock_tokenizer.get_vocab_size.return_value = 8192
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer

            mock_vision = Mock()
            mock_vision_cls.from_pretrained.return_value = mock_vision

            mock_processor = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor

            mock_decoder = Mock()
            mock_decoder.vocab_size = 8192
            mock_load.return_value = (mock_decoder, {'max_caption_len': 12})

            # Mock cv2 to simulate pressing 'q' immediately
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.waitKey.side_effect = [ord('q')]  # Quit immediately
            mock_cv2.getTextSize.return_value = ((100, 20), 5)  # Mock text size

            # Run main
            demo.main()

            # Verify setup was called
            mock_load.assert_called_once()
            mock_cv2.VideoCapture.assert_called_once()

    @patch('src.deployment.jetson_webcam_demo.cv2')
    @patch('src.deployment.jetson_webcam_demo.CLIPVisionModel')
    @patch('src.deployment.jetson_webcam_demo.CLIPImageProcessor')
    @patch('src.deployment.jetson_webcam_demo.Tokenizer')
    @patch('src.deployment.jetson_webcam_demo.load_onnx_model')
    def test_main_onnx_backend(self, mock_load_onnx, mock_tokenizer_cls,
                               mock_processor_cls, mock_vision_cls, mock_cv2):
        """Test main function with ONNX backend."""
        with patch('sys.argv', ['prog', '--onnx', 'test.onnx',
                               '--config', 'config.json',
                               '--tokenizer', 'tok.json']):
            # Mock components
            mock_tokenizer = Mock()
            mock_tokenizer.token_to_id.side_effect = lambda x: {
                "<PAD>": 0, "<MASK>": 3, "<BOS>": 1, "<EOS>": 2
            }.get(x, 5)
            mock_tokenizer.get_vocab_size.return_value = 8192
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer

            mock_vision = Mock()
            mock_vision_cls.from_pretrained.return_value = mock_vision

            mock_processor = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor

            mock_session = Mock()
            mock_load_onnx.return_value = (mock_session, {'max_caption_len': 12})

            # Mock cv2 to quit immediately
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.waitKey.side_effect = [ord('q')]
            mock_cv2.getTextSize.return_value = ((100, 20), 5)  # Mock text size

            # Run main
            demo.main()

            # Verify
            mock_load_onnx.assert_called_once()

    @patch('src.deployment.jetson_webcam_demo.CaptionGenerator')
    @patch('src.deployment.jetson_webcam_demo.cv2')
    @patch('src.deployment.jetson_webcam_demo.CLIPVisionModel')
    @patch('src.deployment.jetson_webcam_demo.CLIPImageProcessor')
    @patch('src.deployment.jetson_webcam_demo.Tokenizer')
    @patch('src.deployment.jetson_webcam_demo.load_pytorch_model')
    def test_main_spacebar_capture(self, mock_load, mock_tokenizer_cls,
                                   mock_processor_cls, mock_vision_cls, mock_cv2,
                                   mock_generator_cls):
        """Test spacebar capture functionality."""
        with patch('sys.argv', ['prog', '--checkpoint', 'test.pt',
                               '--tokenizer', 'tok.json', '--steps', '2']):
            # Mock components
            mock_tokenizer = Mock()
            mock_tokenizer.token_to_id.side_effect = lambda x: {
                "<PAD>": 0, "<MASK>": 3, "<BOS>": 1, "<EOS>": 2
            }.get(x, 5)
            mock_tokenizer.get_vocab_size.return_value = 8192
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer

            mock_vision = Mock()
            mock_vision_cls.from_pretrained.return_value = mock_vision

            mock_processor = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor

            mock_decoder = Mock()
            mock_decoder.vocab_size = 8192
            mock_load.return_value = (mock_decoder, {'max_caption_len': 12})

            # Mock CaptionGenerator
            mock_generator = Mock()
            mock_generator.generate.return_value = ("test caption", 1.5)
            mock_generator_cls.return_value = mock_generator

            # Mock cv2: press spacebar then quit
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            # waitKey: first iteration space pressed, second iteration quit
            mock_cv2.waitKey.side_effect = [ord(' '), ord('q')]
            mock_cv2.getTextSize.return_value = ((100, 20), 5)  # Mock text size

            # Mock time to avoid debounce
            # last_capture_time starts at 0, so current_time must be > 1.0 to pass debounce
            with patch('src.deployment.jetson_webcam_demo.time') as mock_time_module:
                mock_time_module.time.side_effect = [2.0, 10.0]  # First capture at t=2.0 (> 1.0 since 0)
                # Run main
                demo.main()

            # Verify generate was called (caption generated from spacebar)
            assert mock_generator.generate.call_count == 1

    def test_main_missing_args(self):
        """Test main with missing required arguments."""
        with patch('sys.argv', ['prog']):
            with pytest.raises(ValueError, match="Must provide either"):
                demo.main()

    def test_main_onnx_without_config(self):
        """Test main with ONNX but no config."""
        with patch('sys.argv', ['prog', '--onnx', 'test.onnx']):
            with pytest.raises(ValueError, match="Must provide --config with --onnx"):
                demo.main()

    @patch('src.deployment.jetson_webcam_demo.cv2')
    @patch('src.deployment.jetson_webcam_demo.CLIPVisionModel')
    @patch('src.deployment.jetson_webcam_demo.CLIPImageProcessor')
    @patch('src.deployment.jetson_webcam_demo.Tokenizer')
    @patch('src.deployment.jetson_webcam_demo.load_pytorch_model')
    def test_main_camera_not_found(self, mock_load, mock_tokenizer_cls,
                                   mock_processor_cls, mock_vision_cls, mock_cv2):
        """Test main when camera cannot be opened."""
        with patch('sys.argv', ['prog', '--checkpoint', 'test.pt',
                               '--tokenizer', 'tok.json']):
            # Mock components
            mock_tokenizer = Mock()
            mock_tokenizer.token_to_id.side_effect = lambda x: {
                "<PAD>": 0, "<MASK>": 3, "<BOS>": 1, "<EOS>": 2
            }.get(x, 5)
            mock_tokenizer.get_vocab_size.return_value = 8192
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer

            mock_vision = Mock()
            mock_vision_cls.from_pretrained.return_value = mock_vision

            mock_processor = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor

            mock_decoder = Mock()
            mock_decoder.vocab_size = 8192
            mock_load.return_value = (mock_decoder, {'max_caption_len': 12})

            # Mock camera failure
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            # Run main - should exit gracefully
            demo.main()

            # Verify cap.release was not called (never opened)
            mock_cap.release.assert_not_called()

    @patch('src.deployment.jetson_webcam_demo.cv2')
    @patch('src.deployment.jetson_webcam_demo.CLIPVisionModel')
    @patch('src.deployment.jetson_webcam_demo.CLIPImageProcessor')
    @patch('src.deployment.jetson_webcam_demo.Tokenizer')
    @patch('src.deployment.jetson_webcam_demo.load_pytorch_model')
    def test_main_frame_read_failure(self, mock_load, mock_tokenizer_cls,
                                     mock_processor_cls, mock_vision_cls, mock_cv2):
        """Test main when frame read fails."""
        with patch('sys.argv', ['prog', '--checkpoint', 'test.pt',
                               '--tokenizer', 'tok.json']):
            # Mock components
            mock_tokenizer = Mock()
            mock_tokenizer.token_to_id.side_effect = lambda x: {
                "<PAD>": 0, "<MASK>": 3, "<BOS>": 1, "<EOS>": 2
            }.get(x, 5)
            mock_tokenizer.get_vocab_size.return_value = 8192
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer

            mock_vision = Mock()
            mock_vision_cls.from_pretrained.return_value = mock_vision

            mock_processor = Mock()
            mock_processor_cls.from_pretrained.return_value = mock_processor

            mock_decoder = Mock()
            mock_decoder.vocab_size = 8192
            mock_load.return_value = (mock_decoder, {'max_caption_len': 12})

            # Mock camera read failure
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)  # Read failed
            mock_cv2.VideoCapture.return_value = mock_cap

            # Run main - should exit gracefully
            demo.main()

            # Verify cleanup
            mock_cap.release.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_generate_with_zero_steps(self, tmp_path):
        """Test generation with zero steps (edge case)."""
        from src.core.model import ModelConfig, DiffusionTransformer

        config = ModelConfig(vocab_size=8192, max_seq_len=128, d_model=256, n_layers=2, n_heads=4)

        # Create real model for state dict
        model = DiffusionTransformer(config)
        state_dict = model.state_dict()

        checkpoint = {
            'decoder_config': config,
            'decoder_state_dict': state_dict,
            'train_config': {},
        }

        checkpoint_path = tmp_path / "test.pt"
        torch.save(checkpoint, checkpoint_path)

        # This should handle gracefully (though not practical)
        decoder, _ = demo.load_pytorch_model(str(checkpoint_path), "cpu")
        assert decoder is not None

    def test_softmax_numerical_stability(self):
        """Test softmax with large values (numerical stability)."""
        x = np.array([[[1000.0, 1001.0, 999.0]]])
        result = demo.CaptionGenerator._softmax(x)

        # Should not overflow
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_onnx_import_error(self, tmp_path):
        """Test ONNX loading when onnxruntime not installed."""
        # Create mock config and onnx file
        config = {'d_model': 768}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b'dummy')

        # Mock import failure
        import sys
        with patch.dict(sys.modules, {'onnxruntime': None}):
            with pytest.raises((ImportError, AttributeError)):
                demo.load_onnx_model(str(onnx_path), str(config_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
