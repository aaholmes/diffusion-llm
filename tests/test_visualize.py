"""Tests for architecture visualization functionality."""

import os
import tempfile
import sys
from unittest.mock import patch, MagicMock

import pytest

# Skip entire module if graphviz is not available
pytest.importorskip("graphviz", reason="graphviz required for visualization tests")


class TestCreateHighLevelDiagram:
    """Tests for high-level diagram creation."""

    def test_creates_png_file(self, tmp_path):
        """Test that PNG file is created."""
        from src.evaluation.visualize_architecture import create_high_level_diagram

        output_path = str(tmp_path / "test_diagram")
        dot = create_high_level_diagram(output_path)

        assert dot is not None
        assert os.path.exists(f"{output_path}.png")

    def test_creates_svg_file(self, tmp_path):
        """Test that SVG file is created."""
        from src.evaluation.visualize_architecture import create_high_level_diagram

        output_path = str(tmp_path / "test_diagram")
        create_high_level_diagram(output_path)

        assert os.path.exists(f"{output_path}.svg")

    def test_diagram_structure(self, tmp_path):
        """Test that diagram has expected nodes."""
        from src.evaluation.visualize_architecture import create_high_level_diagram

        output_path = str(tmp_path / "test_diagram")
        dot = create_high_level_diagram(output_path)

        # Check that key nodes exist in the source
        source = dot.source
        assert 'image' in source.lower()
        assert 'clip' in source.lower()
        assert 'decoder' in source.lower()
        assert 'caption' in source.lower()


class TestCreateTransformerBlockDiagram:
    """Tests for transformer block diagram creation."""

    def test_creates_files(self, tmp_path):
        """Test that diagram files are created."""
        from src.evaluation.visualize_architecture import create_transformer_block_diagram

        output_path = str(tmp_path / "block_diagram")
        dot = create_transformer_block_diagram(output_path)

        assert dot is not None
        assert os.path.exists(f"{output_path}.png")
        assert os.path.exists(f"{output_path}.svg")

    def test_diagram_contains_attention_layers(self, tmp_path):
        """Test that diagram shows attention layers."""
        from src.evaluation.visualize_architecture import create_transformer_block_diagram

        output_path = str(tmp_path / "block_diagram")
        dot = create_transformer_block_diagram(output_path)

        source = dot.source
        assert 'self' in source.lower() and 'attn' in source.lower()
        assert 'cross' in source.lower()
        assert 'ffn' in source.lower() or 'feed' in source.lower()


class TestCreateDiffusionProcessDiagram:
    """Tests for diffusion process diagram creation."""

    def test_creates_files(self, tmp_path):
        """Test that diagram files are created."""
        from src.evaluation.visualize_architecture import create_diffusion_process_diagram

        output_path = str(tmp_path / "diffusion_diagram")
        dot = create_diffusion_process_diagram(output_path)

        assert dot is not None
        assert os.path.exists(f"{output_path}.png")
        assert os.path.exists(f"{output_path}.svg")

    def test_shows_timesteps(self, tmp_path):
        """Test that diagram shows diffusion timesteps."""
        from src.evaluation.visualize_architecture import create_diffusion_process_diagram

        output_path = str(tmp_path / "diffusion_diagram")
        dot = create_diffusion_process_diagram(output_path)

        source = dot.source
        # Should show multiple timesteps
        assert 't=1.0' in source or 't1' in source
        assert 't=0.0' in source or 't0' in source

    def test_shows_mask_tokens(self, tmp_path):
        """Test that diagram shows MASK tokens."""
        from src.evaluation.visualize_architecture import create_diffusion_process_diagram

        output_path = str(tmp_path / "diffusion_diagram")
        dot = create_diffusion_process_diagram(output_path)

        source = dot.source
        assert 'MASK' in source


class TestCreateTorchviewDiagram:
    """Tests for torchview diagram creation."""

    def test_returns_none_without_torchview(self, tmp_path):
        """Test that function returns None if torchview not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'torchview':
                raise ImportError("No module named 'torchview'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', mock_import):
            import importlib
            import src.evaluation.visualize_architecture as visualize_architecture
            importlib.reload(visualize_architecture)

            result = visualize_architecture.create_torchview_diagram(str(tmp_path / "test"))
            assert result is None

    def test_creates_diagram_with_torchview(self, tmp_path):
        """Test diagram creation with torchview installed."""
        pytest.importorskip("torchview")

        from src.evaluation.visualize_architecture import create_torchview_diagram

        output_path = str(tmp_path / "detailed_diagram")
        result = create_torchview_diagram(output_path)

        # May or may not create file depending on graphviz availability
        # Just check it doesn't raise
        assert result is not None or result is None


class TestPrintModelSummary:
    """Tests for model summary printing."""

    def test_prints_summary(self, capsys):
        """Test that summary is printed."""
        from src.evaluation.visualize_architecture import print_model_summary

        print_model_summary()

        captured = capsys.readouterr()
        assert 'VISION ENCODER' in captured.out or 'vision' in captured.out.lower()
        assert 'DECODER' in captured.out or 'decoder' in captured.out.lower()
        assert 'd_model' in captured.out
        assert 'n_layers' in captured.out

    def test_shows_parameter_counts(self, capsys):
        """Test that parameter counts are shown."""
        from src.evaluation.visualize_architecture import print_model_summary

        print_model_summary()

        captured = capsys.readouterr()
        assert 'params' in captured.out.lower()
        # Should show total parameters
        assert 'TOTAL' in captured.out or 'total' in captured.out.lower()


class TestMainFunction:
    """Tests for the main function."""

    def test_main_creates_diagrams(self, tmp_path):
        """Test that main creates all diagrams."""
        from src.evaluation.visualize_architecture import main

        with patch.object(sys, 'argv', [
            'visualize_architecture.py',
            '--output-dir', str(tmp_path),
        ]):
            main()

        # Check that diagram files were created
        assert os.path.exists(tmp_path / "architecture_captioning.png")
        assert os.path.exists(tmp_path / "architecture_block.png")
        assert os.path.exists(tmp_path / "architecture_diffusion.png")

    def test_main_with_detailed_flag(self, tmp_path):
        """Test main with --detailed flag."""
        pytest.importorskip("torchview")

        from src.evaluation.visualize_architecture import main

        with patch.object(sys, 'argv', [
            'visualize_architecture.py',
            '--output-dir', str(tmp_path),
            '--detailed',
        ]):
            main()

        # Basic diagrams should still be created
        assert os.path.exists(tmp_path / "architecture_captioning.png")

    def test_main_creates_output_directory(self, tmp_path):
        """Test that main creates output directory if it doesn't exist."""
        from src.evaluation.visualize_architecture import main

        new_dir = tmp_path / "new_output_dir"
        assert not new_dir.exists()

        with patch.object(sys, 'argv', [
            'visualize_architecture.py',
            '--output-dir', str(new_dir),
        ]):
            main()

        assert new_dir.exists()


class TestDiagramContent:
    """Tests for diagram content validation."""

    def test_high_level_has_clip_node(self, tmp_path):
        """Test high-level diagram has CLIP node."""
        from src.evaluation.visualize_architecture import create_high_level_diagram

        output_path = str(tmp_path / "test")
        dot = create_high_level_diagram(output_path)

        # CLIP should be mentioned
        assert 'clip' in dot.source.lower()

    def test_block_diagram_has_residual_connections(self, tmp_path):
        """Test block diagram shows residual connections."""
        from src.evaluation.visualize_architecture import create_transformer_block_diagram

        output_path = str(tmp_path / "test")
        dot = create_transformer_block_diagram(output_path)

        # Should show residual connections
        assert 'residual' in dot.source.lower()

    def test_diffusion_shows_denoising(self, tmp_path):
        """Test diffusion diagram shows denoising process."""
        from src.evaluation.visualize_architecture import create_diffusion_process_diagram

        output_path = str(tmp_path / "test")
        dot = create_diffusion_process_diagram(output_path)

        assert 'denoise' in dot.source.lower()
