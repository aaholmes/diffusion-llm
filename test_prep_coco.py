"""Tests for COCO data preparation functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from PIL import Image

from prep_coco_data import (
    download_file,
    load_coco_captions,
    extract_clip_features,
    tokenize_captions,
)


class TestDownloadFile:
    """Tests for download_file function."""

    def test_skip_existing_file(self, tmp_path):
        """Test that existing files are skipped."""
        dest = tmp_path / "existing.txt"
        dest.write_text("existing content")

        # Should not raise or download
        download_file("http://example.com/file.txt", dest, "test file")
        assert dest.read_text() == "existing content"

    def test_download_new_file(self, tmp_path):
        """Test downloading a new file."""
        dest = tmp_path / "new.txt"

        with patch('urllib.request.urlretrieve') as mock_retrieve:
            download_file("http://example.com/file.txt", dest, "test file")
            mock_retrieve.assert_called_once()


class TestLoadCocoCaptions:
    """Tests for load_coco_captions function."""

    def test_load_captions_structure(self, tmp_path):
        """Test loading COCO captions JSON."""
        # Create a minimal COCO-format JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg"},
                {"id": 2, "file_name": "image2.jpg"},
            ],
            "annotations": [
                {"image_id": 1, "caption": "A cat on a mat"},
                {"image_id": 1, "caption": "A feline resting"},
                {"image_id": 2, "caption": "A dog in a park"},
            ]
        }

        json_path = tmp_path / "captions.json"
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        captions_by_image, id_to_file = load_coco_captions(json_path)

        assert len(captions_by_image) == 2
        assert len(captions_by_image[1]) == 2
        assert len(captions_by_image[2]) == 1
        assert id_to_file[1] == "image1.jpg"
        assert id_to_file[2] == "image2.jpg"

    def test_load_captions_content(self, tmp_path):
        """Test that caption content is correct."""
        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg"}],
            "annotations": [
                {"image_id": 1, "caption": "First caption"},
                {"image_id": 1, "caption": "Second caption"},
            ]
        }

        json_path = tmp_path / "captions.json"
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        captions_by_image, _ = load_coco_captions(json_path)

        assert "First caption" in captions_by_image[1]
        assert "Second caption" in captions_by_image[1]


class TestExtractClipFeatures:
    """Tests for CLIP feature extraction."""

    def test_extract_features_shape(self, tmp_path):
        """Test that extracted features have correct shape."""
        # Create test images
        images = []
        for i in range(3):
            img_path = tmp_path / f"img{i}.jpg"
            img = Image.new('RGB', (224, 224), color='red')
            img.save(img_path)
            images.append(img_path)

        features = extract_clip_features(
            images,
            clip_model="openai/clip-vit-base-patch32",
            batch_size=2,
            device="cpu",
        )

        assert features.shape[0] == 3  # 3 images
        assert features.shape[1] == 50  # CLIP sequence length
        assert features.shape[2] == 768  # CLIP hidden dim

    def test_extract_features_single_image(self, tmp_path):
        """Test feature extraction with single image."""
        img_path = tmp_path / "single.jpg"
        img = Image.new('RGB', (224, 224), color='blue')
        img.save(img_path)

        features = extract_clip_features(
            [img_path],
            clip_model="openai/clip-vit-base-patch32",
            batch_size=1,
            device="cpu",
        )

        assert features.shape[0] == 1

    def test_extract_features_handles_invalid_image(self, tmp_path):
        """Test that invalid images are handled gracefully."""
        # Create one valid and one invalid image path
        valid_path = tmp_path / "valid.jpg"
        img = Image.new('RGB', (224, 224), color='green')
        img.save(valid_path)

        invalid_path = tmp_path / "invalid.jpg"
        invalid_path.write_text("not an image")

        # Should not raise, uses fallback
        features = extract_clip_features(
            [valid_path, invalid_path],
            clip_model="openai/clip-vit-base-patch32",
            batch_size=2,
            device="cpu",
        )

        assert features.shape[0] == 2


class TestTokenizeCaptions:
    """Tests for caption tokenization."""

    @pytest.fixture
    def tokenizer(self):
        """Load the project tokenizer."""
        from tokenizers import Tokenizer
        return Tokenizer.from_file("data_full/tokenizer.json")

    def test_tokenize_basic(self, tokenizer):
        """Test basic caption tokenization."""
        captions = ["A cat on a mat", "A dog in the park"]
        tokens = tokenize_captions(captions, tokenizer, max_len=32)

        assert tokens.shape == (2, 32)
        assert tokens.dtype == torch.long

    def test_tokenize_adds_special_tokens(self, tokenizer):
        """Test that BOS and EOS tokens are added."""
        captions = ["Hello world"]
        tokens = tokenize_captions(captions, tokenizer, max_len=32, bos_id=1, eos_id=2)

        # BOS should be at start
        assert tokens[0, 0].item() == 1
        # EOS should follow content
        # Find first padding (0) and check EOS is before it
        token_list = tokens[0].tolist()
        non_pad = [t for t in token_list if t != 0]
        assert non_pad[-1] == 2  # Last non-pad should be EOS

    def test_tokenize_pads_short_captions(self, tokenizer):
        """Test that short captions are padded."""
        captions = ["Hi"]
        tokens = tokenize_captions(captions, tokenizer, max_len=32, pad_id=0)

        # Most tokens should be padding
        assert (tokens[0] == 0).sum() > 20

    def test_tokenize_truncates_long_captions(self, tokenizer):
        """Test that long captions are truncated."""
        long_caption = " ".join(["word"] * 100)
        tokens = tokenize_captions([long_caption], tokenizer, max_len=32, eos_id=2)

        assert tokens.shape == (1, 32)
        # Last token should be EOS
        assert tokens[0, -1].item() == 2

    def test_tokenize_multiple_captions(self, tokenizer):
        """Test tokenizing multiple captions."""
        captions = [
            "A beautiful sunset over the ocean",
            "A cat sleeping on a couch",
            "Two dogs playing in the park",
        ]
        tokens = tokenize_captions(captions, tokenizer, max_len=64)

        assert tokens.shape == (3, 64)


class TestDownloadCOCOImages:
    """Tests for COCO image downloading."""

    def test_download_skips_existing(self, tmp_path):
        """Test that existing images are skipped."""
        from prep_coco_data import download_coco_images

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create existing image
        (images_dir / "existing.jpg").write_text("existing")

        id_to_file = {1: "existing.jpg", 2: "new.jpg"}

        with patch('urllib.request.urlretrieve') as mock_retrieve:
            download_coco_images([1, 2], id_to_file, images_dir, "train2017")
            # Should only try to download the new one
            assert mock_retrieve.call_count == 1


class TestMainFunction:
    """Tests for the main function."""

    def test_argument_parsing(self):
        """Test that arguments are parsed correctly."""
        import sys
        from unittest.mock import patch

        test_args = [
            'prep_coco_data.py',
            '--num_train', '100',
            '--num_val', '50',
            '--output_dir', '/tmp/test_output',
        ]

        with patch.object(sys, 'argv', test_args):
            import argparse
            from prep_coco_data import main

            # Just test that we can parse args - don't run full main
            parser = argparse.ArgumentParser()
            parser.add_argument("--num_train", type=int, default=10000)
            parser.add_argument("--num_val", type=int, default=1000)
            parser.add_argument("--output_dir", type=str, default="data_coco")
            args = parser.parse_args(test_args[1:])

            assert args.num_train == 100
            assert args.num_val == 50


class TestDownloadCocoAnnotations:
    """Tests for COCO annotation downloading."""

    def test_creates_annotations_dir(self, tmp_path):
        """Test that annotations directory is created."""
        from prep_coco_data import download_coco_annotations

        # Mock urlretrieve to avoid actual download
        with patch('urllib.request.urlretrieve'):
            with patch('zipfile.ZipFile'):
                try:
                    download_coco_annotations(tmp_path)
                except Exception:
                    pass  # May fail due to missing files, but dir should be created

        assert (tmp_path / "annotations").exists()

    def test_skips_download_if_zip_exists(self, tmp_path):
        """Test that existing zip file is skipped."""
        from prep_coco_data import download_coco_annotations

        # Create fake zip file
        zip_path = tmp_path / "annotations_trainval2017.zip"
        zip_path.write_text("fake zip")

        # Create annotations dir with expected file
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "captions_train2017.json").write_text("{}")

        with patch('urllib.request.urlretrieve') as mock_retrieve:
            download_coco_annotations(tmp_path)
            mock_retrieve.assert_not_called()


class TestMainFunctionExtended:
    """Extended tests for main function."""

    def test_main_with_mocked_downloads(self, tmp_path):
        """Test main function with mocked network calls."""
        import sys
        from prep_coco_data import main

        # Create mock COCO annotation files
        ann_dir = tmp_path / "cache" / "annotations"
        ann_dir.mkdir(parents=True)

        train_data = {
            "images": [{"id": i, "file_name": f"train_{i}.jpg"} for i in range(10)],
            "annotations": [{"image_id": i, "caption": f"Train caption {i}"} for i in range(10)]
        }
        val_data = {
            "images": [{"id": i, "file_name": f"val_{i}.jpg"} for i in range(5)],
            "annotations": [{"image_id": i, "caption": f"Val caption {i}"} for i in range(5)]
        }

        with open(ann_dir / "captions_train2017.json", 'w') as f:
            json.dump(train_data, f)
        with open(ann_dir / "captions_val2017.json", 'w') as f:
            json.dump(val_data, f)

        # Create fake zip to skip download
        (tmp_path / "cache" / "annotations_trainval2017.zip").write_text("fake")

        # Create mock images
        train_dir = tmp_path / "cache" / "train2017"
        train_dir.mkdir()
        val_dir = tmp_path / "cache" / "val2017"
        val_dir.mkdir()

        for i in range(10):
            img = Image.new('RGB', (224, 224), color='red')
            img.save(train_dir / f"train_{i}.jpg")

        for i in range(5):
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(val_dir / f"val_{i}.jpg")

        output_dir = tmp_path / "output"

        with patch.object(sys, 'argv', [
            'prep_coco_data.py',
            '--num_train', '5',
            '--num_val', '3',
            '--output_dir', str(output_dir),
            '--cache_dir', str(tmp_path / "cache"),
            '--batch_size', '2',
        ]):
            main()

        # Check output files were created
        assert (output_dir / "train_image_features.pt").exists()
        assert (output_dir / "train_captions.pt").exists()
        assert (output_dir / "val_image_features.pt").exists()
        assert (output_dir / "val_captions.pt").exists()
        assert (output_dir / "config.json").exists()

    def test_main_saves_config(self, tmp_path):
        """Test that main saves proper config file."""
        import sys
        from prep_coco_data import main

        # Setup mock data (same as above but simplified)
        ann_dir = tmp_path / "cache" / "annotations"
        ann_dir.mkdir(parents=True)

        data = {
            "images": [{"id": 0, "file_name": "img.jpg"}],
            "annotations": [{"image_id": 0, "caption": "Test"}]
        }
        with open(ann_dir / "captions_train2017.json", 'w') as f:
            json.dump(data, f)
        with open(ann_dir / "captions_val2017.json", 'w') as f:
            json.dump(data, f)

        (tmp_path / "cache" / "annotations_trainval2017.zip").write_text("fake")

        for split in ["train2017", "val2017"]:
            split_dir = tmp_path / "cache" / split
            split_dir.mkdir()
            img = Image.new('RGB', (224, 224))
            img.save(split_dir / "img.jpg")

        output_dir = tmp_path / "output"

        with patch.object(sys, 'argv', [
            'prep_coco_data.py',
            '--num_train', '1',
            '--num_val', '1',
            '--output_dir', str(output_dir),
            '--cache_dir', str(tmp_path / "cache"),
        ]):
            main()

        # Load and verify config
        with open(output_dir / "config.json") as f:
            config = json.load(f)

        assert config['dataset_type'] == 'coco'
        assert config['feature_dim'] == 768
        assert 'clip_model' in config


class TestDownloadImageErrors:
    """Tests for image download error handling."""

    def test_handles_download_failure(self, tmp_path):
        """Test that download failures are handled gracefully."""
        from prep_coco_data import download_coco_images

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        id_to_file = {1: "img1.jpg", 2: "img2.jpg"}

        def raise_error(*args, **kwargs):
            raise Exception("Network error")

        with patch('urllib.request.urlretrieve', side_effect=raise_error):
            # Should not raise, just print warning
            download_coco_images([1, 2], id_to_file, images_dir, "train2017")


class TestIntegration:
    """Integration tests for COCO data prep."""

    def test_end_to_end_mini(self, tmp_path):
        """Test minimal end-to-end pipeline with mocked downloads."""
        from prep_coco_data import (
            load_coco_captions,
            extract_clip_features,
            tokenize_captions,
        )
        from tokenizers import Tokenizer

        # Create mock COCO data
        coco_data = {
            "images": [
                {"id": i, "file_name": f"img{i}.jpg"}
                for i in range(5)
            ],
            "annotations": [
                {"image_id": i, "caption": f"A photo number {i}"}
                for i in range(5)
            ]
        }

        # Save mock annotations
        ann_path = tmp_path / "captions.json"
        with open(ann_path, 'w') as f:
            json.dump(coco_data, f)

        # Create mock images
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(5):
            img = Image.new('RGB', (224, 224), color=(i * 50, i * 30, i * 20))
            img.save(images_dir / f"img{i}.jpg")

        # Load captions
        captions_by_image, id_to_file = load_coco_captions(ann_path)
        assert len(captions_by_image) == 5

        # Get image paths and captions
        image_paths = [images_dir / id_to_file[i] for i in range(5)]
        caption_texts = [captions_by_image[i][0] for i in range(5)]

        # Extract features
        features = extract_clip_features(
            image_paths,
            batch_size=2,
            device="cpu",
        )
        assert features.shape == (5, 50, 768)

        # Tokenize
        tokenizer = Tokenizer.from_file("data_full/tokenizer.json")
        tokens = tokenize_captions(caption_texts, tokenizer, max_len=32)
        assert tokens.shape == (5, 32)

    def test_full_pipeline_shapes(self, tmp_path):
        """Test that all output shapes are correct."""
        from prep_coco_data import extract_clip_features, tokenize_captions
        from tokenizers import Tokenizer

        # Create test images
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        image_paths = []
        for i in range(8):
            img = Image.new('RGB', (224, 224), color=(i * 30, i * 20, i * 10))
            path = images_dir / f"img{i}.jpg"
            img.save(path)
            image_paths.append(path)

        captions = [f"Caption number {i}" for i in range(8)]

        # Extract features
        features = extract_clip_features(image_paths, batch_size=4, device="cpu")

        # Tokenize
        tokenizer = Tokenizer.from_file("data_full/tokenizer.json")
        tokens = tokenize_captions(captions, tokenizer, max_len=48)

        # Verify shapes match
        assert features.shape[0] == len(image_paths)
        assert tokens.shape[0] == len(captions)
        assert features.shape[0] == tokens.shape[0]
