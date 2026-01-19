"""Tests for ONNX export functionality."""

import os
import tempfile
import pytest
import torch

# Skip entire module if onnxscript is not available (required by torch.onnx.export)
pytest.importorskip("onnxscript", reason="onnxscript required for ONNX export tests")

from src.core.model import DiffusionTransformer, ModelConfig
from src.deployment.export_onnx import (
    DecoderWrapper,
    load_decoder,
    export_to_onnx,
    verify_onnx,
    benchmark_onnx,
)


@pytest.fixture
def tiny_config():
    """Create a tiny model config for testing."""
    return ModelConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        vocab_size=100,
        max_seq_len=32,
        has_cross_attention=True,
    )


@pytest.fixture
def tiny_decoder(tiny_config):
    """Create a tiny decoder for testing."""
    return DiffusionTransformer(tiny_config)


@pytest.fixture
def checkpoint_path(tiny_decoder, tiny_config):
    """Create a temporary checkpoint file."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save({
            'decoder_config': {
                'd_model': tiny_config.d_model,
                'n_heads': tiny_config.n_heads,
                'n_layers': tiny_config.n_layers,
                'd_ff': tiny_config.d_ff,
                'vocab_size': tiny_config.vocab_size,
                'max_seq_len': tiny_config.max_seq_len,
                'has_cross_attention': tiny_config.has_cross_attention,
            },
            'decoder_state_dict': tiny_decoder.state_dict(),
        }, f.name)
        yield f.name
    os.unlink(f.name)


class TestDecoderWrapper:
    """Tests for the DecoderWrapper class."""

    def test_wrapper_forward(self, tiny_decoder):
        """Test wrapper forward pass."""
        wrapper = DecoderWrapper(tiny_decoder)
        wrapper.eval()

        batch_size = 2
        seq_len = 16
        encoder_seq_len = 10

        x = torch.randint(0, 100, (batch_size, seq_len))
        t = torch.rand(batch_size)
        encoder_output = torch.randn(batch_size, encoder_seq_len, 64)

        with torch.no_grad():
            logits = wrapper(x, t, encoder_output)

        assert logits.shape == (batch_size, seq_len, 100)

    def test_wrapper_single_batch(self, tiny_decoder):
        """Test wrapper with batch size 1."""
        wrapper = DecoderWrapper(tiny_decoder)
        wrapper.eval()

        x = torch.randint(0, 100, (1, 16))
        t = torch.rand(1)
        encoder_output = torch.randn(1, 10, 64)

        with torch.no_grad():
            logits = wrapper(x, t, encoder_output)

        assert logits.shape == (1, 16, 100)

    def test_wrapper_creates_attention_mask(self, tiny_decoder):
        """Test that wrapper creates proper attention mask."""
        wrapper = DecoderWrapper(tiny_decoder)

        x = torch.randint(0, 100, (2, 16))
        t = torch.rand(2)
        encoder_output = torch.randn(2, 10, 64)

        # The wrapper should create an attention mask internally
        with torch.no_grad():
            logits = wrapper(x, t, encoder_output)

        # If no error, mask was created correctly
        assert logits is not None


class TestLoadDecoder:
    """Tests for load_decoder function."""

    def test_load_decoder_from_checkpoint(self, checkpoint_path):
        """Test loading decoder from checkpoint."""
        decoder, config = load_decoder(checkpoint_path)

        assert isinstance(decoder, DiffusionTransformer)
        assert isinstance(config, ModelConfig)
        assert config.d_model == 64
        assert config.n_layers == 2

    def test_load_decoder_eval_mode(self, checkpoint_path):
        """Test that loaded decoder is in eval mode."""
        decoder, _ = load_decoder(checkpoint_path)
        assert not decoder.training

    def test_load_decoder_device(self, checkpoint_path):
        """Test loading decoder to specific device."""
        decoder, _ = load_decoder(checkpoint_path, device="cpu")
        # Check parameters are on CPU
        for param in decoder.parameters():
            assert param.device.type == "cpu"
            break


class TestExportToOnnx:
    """Tests for ONNX export functionality."""

    def test_export_creates_file(self, tiny_decoder, tiny_config):
        """Test that export creates an ONNX file."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            export_to_onnx(
                tiny_decoder, tiny_config, output_path,
                batch_size=1, seq_len=16, encoder_seq_len=10,
            )
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_different_seq_lengths(self, tiny_decoder, tiny_config):
        """Test export with different sequence lengths."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            export_to_onnx(
                tiny_decoder, tiny_config, output_path,
                batch_size=1, seq_len=32, encoder_seq_len=20,
            )
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestVerifyOnnx:
    """Tests for ONNX verification."""

    def test_verify_onnx_success(self, tiny_decoder, tiny_config):
        """Test ONNX verification succeeds for valid model."""
        pytest.importorskip("onnxruntime")

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            export_to_onnx(
                tiny_decoder, tiny_config, output_path,
                batch_size=1, seq_len=16, encoder_seq_len=10,
            )
            result = verify_onnx(output_path, tiny_config, batch_size=1, seq_len=16, encoder_seq_len=10)
            assert result is True
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_verify_onnx_output_shape(self, tiny_decoder, tiny_config):
        """Test that ONNX model produces correct output shape."""
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            export_to_onnx(
                tiny_decoder, tiny_config, output_path,
                batch_size=1, seq_len=16, encoder_seq_len=10,
            )

            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

            x = np.random.randint(0, 100, (1, 16)).astype(np.int64)
            t = np.random.rand(1).astype(np.float32)
            encoder_output = np.random.randn(1, 10, 64).astype(np.float32)

            outputs = session.run(None, {
                'tokens': x,
                'timestep': t,
                'image_features': encoder_output,
            })

            assert outputs[0].shape == (1, 16, 100)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestBenchmarkOnnx:
    """Tests for ONNX benchmarking."""

    def test_benchmark_runs(self, tiny_decoder, tiny_config):
        """Test that benchmark runs without error."""
        pytest.importorskip("onnxruntime")

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            export_to_onnx(
                tiny_decoder, tiny_config, output_path,
                batch_size=1, seq_len=16, encoder_seq_len=10,
            )
            # Run with minimal iterations
            benchmark_onnx(output_path, tiny_config, num_runs=5, seq_len=16, encoder_seq_len=10)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMainFunction:
    """Tests for the main function."""

    def test_main_export_only(self, checkpoint_path):
        """Test main function with export only."""
        import sys
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            with patch.object(sys, 'argv', [
                'export_onnx.py',
                '--checkpoint', checkpoint_path,
                '--output', output_path,
                '--seq_len', '16',
            ]):
                from src.deployment.export_onnx import main
                main()

            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_main_with_test_flag(self, checkpoint_path):
        """Test main function with --test flag."""
        pytest.importorskip("onnxruntime")
        import sys
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name

        try:
            with patch.object(sys, 'argv', [
                'export_onnx.py',
                '--checkpoint', checkpoint_path,
                '--output', output_path,
                '--seq_len', '16',
                '--test',
            ]):
                from src.deployment.export_onnx import main
                main()

            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestOnnxRuntimeNotInstalled:
    """Tests for handling missing onnxruntime."""

    def test_verify_without_onnxruntime(self, tiny_config, monkeypatch):
        """Test verify_onnx when onnxruntime is not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ImportError("No module named 'onnxruntime'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)

        # Re-import to get fresh function
        import importlib
        import src.deployment.export_onnx as export_onnx
        importlib.reload(export_onnx)

        result = export_onnx.verify_onnx("dummy.onnx", tiny_config)
        assert result is False

    def test_benchmark_without_onnxruntime(self, tiny_config, monkeypatch):
        """Test benchmark_onnx when onnxruntime is not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ImportError("No module named 'onnxruntime'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)

        import importlib
        import src.deployment.export_onnx as export_onnx
        importlib.reload(export_onnx)

        # Should not raise, just return
        export_onnx.benchmark_onnx("dummy.onnx", tiny_config)
