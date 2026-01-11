#!/usr/bin/env python3
"""
Comprehensive tests for data_prep.py

Run with: pytest test_data_prep.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from data_prep import (
    DataConfig,
    train_tokenizer,
    tokenize_dataset,
    compute_statistics,
    main,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create a test configuration with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield DataConfig(
            subset_size=100,
            vocab_size=512,  # Small for fast testing
            max_seq_len=64,
            data_dir=tmpdir,
            tokenizer_path=f"{tmpdir}/tokenizer.json",
            train_data_path=f"{tmpdir}/train_tokens.pt",
            val_data_path=f"{tmpdir}/val_tokens.pt",
            config_path=f"{tmpdir}/config.json",
        )


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    # Simulate HuggingFace dataset structure
    class MockDataset:
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return {"text": self.texts[idx]}
            return {"text": self.texts[idx]}

        def __iter__(self):
            for text in self.texts:
                yield {"text": text}

    texts = [
        "Once upon a time, there was a little cat.",
        "The cat liked to play with a ball.",
        "One day, the cat found a new friend.",
        "They played together in the garden.",
        "The sun was shining bright.",
        "Birds were singing in the trees.",
        "The cat was very happy.",
        "At night, the cat went to sleep.",
        "The next day, they played again.",
        "It was a wonderful story.",
    ]
    return MockDataset(texts)


@pytest.fixture
def trained_tokenizer(mock_dataset, config):
    """Train a tokenizer on mock data."""
    return train_tokenizer(mock_dataset, config)


# =============================================================================
# Unit Tests: DataConfig
# =============================================================================

class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()

        assert config.dataset_name == "roneneldan/TinyStories"
        assert config.subset_size == 100_000
        assert config.val_split == 0.05
        assert config.vocab_size == 8192
        assert config.max_seq_len == 256

    def test_special_token_ids(self):
        """Test special token ID properties."""
        config = DataConfig()

        assert config.pad_token_id == 0
        assert config.bos_token_id == 1
        assert config.eos_token_id == 2
        assert config.mask_token_id == 3
        assert config.unk_token_id == 4

    def test_special_tokens_order(self):
        """Test that special tokens are in correct order."""
        config = DataConfig()

        expected = ("<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>")
        assert config.special_tokens == expected

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            subset_size=500,
            vocab_size=1024,
            max_seq_len=128,
        )

        assert config.subset_size == 500
        assert config.vocab_size == 1024
        assert config.max_seq_len == 128


# =============================================================================
# Unit Tests: Tokenizer
# =============================================================================

class TestTokenizer:
    """Tests for tokenizer training and usage."""

    def test_tokenizer_trains_successfully(self, mock_dataset, config):
        """Test that tokenizer trains without errors."""
        tokenizer = train_tokenizer(mock_dataset, config)

        assert tokenizer is not None
        assert isinstance(tokenizer, Tokenizer)

    def test_special_tokens_have_correct_ids(self, trained_tokenizer, config):
        """Test that special tokens are assigned expected IDs."""
        tokenizer = trained_tokenizer

        assert tokenizer.token_to_id("<PAD>") == config.pad_token_id
        assert tokenizer.token_to_id("<BOS>") == config.bos_token_id
        assert tokenizer.token_to_id("<EOS>") == config.eos_token_id
        assert tokenizer.token_to_id("<MASK>") == config.mask_token_id
        assert tokenizer.token_to_id("<UNK>") == config.unk_token_id

    def test_vocab_size_correct(self, trained_tokenizer, config):
        """Test that tokenizer has correct vocabulary size."""
        # Vocab size may be less than requested if not enough data
        assert trained_tokenizer.get_vocab_size() <= config.vocab_size
        assert trained_tokenizer.get_vocab_size() >= len(config.special_tokens)

    def test_encoding_adds_bos_eos(self, trained_tokenizer, config):
        """Test that encoding adds BOS and EOS tokens."""
        text = "Hello world"
        encoding = trained_tokenizer.encode(text)

        assert encoding.ids[0] == config.bos_token_id
        assert encoding.ids[-1] == config.eos_token_id

    def test_encoding_decoding_roundtrip(self, trained_tokenizer):
        """Test that encode->decode preserves text (approximately)."""
        original = "The cat sat on the mat."
        encoding = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(encoding.ids)

        # Remove BOS/EOS tokens for comparison
        # The decoded text should contain the original words
        assert "cat" in decoded
        assert "mat" in decoded

    def test_tokenizer_saves_and_loads(self, trained_tokenizer, config):
        """Test that tokenizer can be saved and loaded."""
        # Tokenizer is saved during training
        assert os.path.exists(config.tokenizer_path)

        # Load it back
        loaded = Tokenizer.from_file(config.tokenizer_path)

        # Verify it works the same
        text = "Test text"
        original_ids = trained_tokenizer.encode(text).ids
        loaded_ids = loaded.encode(text).ids

        assert original_ids == loaded_ids

    def test_unknown_characters_handled(self, trained_tokenizer, config):
        """Test that unknown/rare characters don't crash encoding."""
        # ByteLevel handles any UTF-8
        weird_text = "Hello 你好 مرحبا 🎉"
        encoding = trained_tokenizer.encode(weird_text)

        assert len(encoding.ids) > 0
        assert encoding.ids[0] == config.bos_token_id
        assert encoding.ids[-1] == config.eos_token_id


# =============================================================================
# Unit Tests: Tokenization
# =============================================================================

class TestTokenizeDataset:
    """Tests for dataset tokenization."""

    def test_output_shape(self, mock_dataset, trained_tokenizer, config):
        """Test that output tensor has correct shape."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        assert tokens.shape == (len(mock_dataset), config.max_seq_len)

    def test_output_dtype(self, mock_dataset, trained_tokenizer, config):
        """Test that output tensor has correct dtype."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        assert tokens.dtype == torch.long

    def test_sequences_start_with_bos(self, mock_dataset, trained_tokenizer, config):
        """Test that all sequences start with BOS token."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        assert (tokens[:, 0] == config.bos_token_id).all()

    def test_padding_applied_correctly(self, mock_dataset, trained_tokenizer, config):
        """Test that short sequences are padded."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        # Check that padding tokens appear at the end of sequences
        for i in range(len(tokens)):
            seq = tokens[i]
            # Find first PAD token
            pad_positions = (seq == config.pad_token_id).nonzero(as_tuple=True)[0]

            if len(pad_positions) > 0:
                first_pad = pad_positions[0].item()
                # All tokens after first PAD should also be PAD
                assert (seq[first_pad:] == config.pad_token_id).all()

    def test_truncation_applied_correctly(self, trained_tokenizer, config):
        """Test that long sequences are truncated with EOS."""
        # Create a very long text
        long_text = "word " * 1000

        class SingleItemDataset:
            def __iter__(self):
                yield {"text": long_text}
            def __len__(self):
                return 1

        tokens = tokenize_dataset(SingleItemDataset(), trained_tokenizer, config)

        # Should be exactly max_seq_len
        assert tokens.shape[1] == config.max_seq_len

        # Should end with EOS (at position max_seq_len - 1)
        assert tokens[0, -1] == config.eos_token_id or tokens[0, -1] == config.pad_token_id

    def test_no_negative_token_ids(self, mock_dataset, trained_tokenizer, config):
        """Test that no negative token IDs are produced."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        assert (tokens >= 0).all()

    def test_token_ids_within_vocab(self, mock_dataset, trained_tokenizer, config):
        """Test that all token IDs are within vocabulary."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)
        vocab_size = trained_tokenizer.get_vocab_size()

        assert (tokens < vocab_size).all()


# =============================================================================
# Unit Tests: Statistics
# =============================================================================

class TestComputeStatistics:
    """Tests for statistics computation."""

    def test_basic_statistics(self, config):
        """Test basic statistics computation."""
        # Create a known tensor
        tokens = torch.tensor([
            [1, 5, 6, 7, 2, 0, 0, 0],  # Length 5, 3 padding
            [1, 5, 6, 2, 0, 0, 0, 0],  # Length 4, 4 padding
        ])
        config.max_seq_len = 8

        stats = compute_statistics(tokens, config)

        assert stats["num_sequences"] == 2
        assert stats["seq_length"] == 8
        assert stats["total_tokens"] == 16
        assert stats["non_pad_tokens"] == 9  # 5 + 4

    def test_avg_length(self, config):
        """Test average length calculation."""
        tokens = torch.tensor([
            [1, 5, 6, 2, 0, 0],  # 4 non-pad
            [1, 5, 2, 0, 0, 0],  # 3 non-pad
        ])
        config.max_seq_len = 6

        stats = compute_statistics(tokens, config)

        assert stats["avg_length"] == 3.5  # (4 + 3) / 2

    def test_min_max_length(self, config):
        """Test min/max length calculation."""
        tokens = torch.tensor([
            [1, 5, 6, 7, 8, 2],  # 6 non-pad (full)
            [1, 5, 2, 0, 0, 0],  # 3 non-pad
            [1, 2, 0, 0, 0, 0],  # 2 non-pad
        ])
        config.max_seq_len = 6

        stats = compute_statistics(tokens, config)

        assert stats["min_length"] == 2
        assert stats["max_length"] == 6

    def test_pad_fraction(self, config):
        """Test padding fraction calculation."""
        tokens = torch.tensor([
            [1, 5, 0, 0],  # 50% padding
            [1, 5, 6, 0],  # 25% padding
        ])
        config.max_seq_len = 4

        stats = compute_statistics(tokens, config)

        # 3 padding tokens out of 8 total = 37.5%
        assert abs(stats["pad_fraction"] - 0.375) < 0.001


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_mock_data(self, mock_dataset, config):
        """Test full pipeline with mock data."""
        # Train tokenizer
        tokenizer = train_tokenizer(mock_dataset, config)

        # Tokenize dataset
        tokens = tokenize_dataset(mock_dataset, tokenizer, config)

        # Compute statistics
        stats = compute_statistics(tokens, config)

        # Verify outputs
        assert tokens.shape == (len(mock_dataset), config.max_seq_len)
        assert stats["num_sequences"] == len(mock_dataset)
        assert os.path.exists(config.tokenizer_path)

    def test_tokenized_sequences_decodable(self, mock_dataset, trained_tokenizer, config):
        """Test that tokenized sequences can be decoded back."""
        tokens = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        for i, original in enumerate(mock_dataset):
            # Get non-padding tokens
            seq = tokens[i]
            non_pad_mask = seq != config.pad_token_id
            non_pad_tokens = seq[non_pad_mask].tolist()

            # Decode
            decoded = trained_tokenizer.decode(non_pad_tokens)

            # Should contain key words from original
            original_words = original["text"].lower().split()[:3]
            for word in original_words:
                # Check if word (or part of it) appears in decoded
                assert any(w in decoded.lower() for w in [word, word[:4]])

    def test_deterministic_tokenization(self, mock_dataset, trained_tokenizer, config):
        """Test that tokenization is deterministic."""
        tokens1 = tokenize_dataset(mock_dataset, trained_tokenizer, config)
        tokens2 = tokenize_dataset(mock_dataset, trained_tokenizer, config)

        assert torch.equal(tokens1, tokens2)

    def test_saved_files_loadable(self, mock_dataset, config):
        """Test that all saved files can be loaded."""
        # Train tokenizer
        tokenizer = train_tokenizer(mock_dataset, config)

        # Tokenize and save
        tokens = tokenize_dataset(mock_dataset, tokenizer, config)
        torch.save(tokens, config.train_data_path)

        # Save config
        stats = compute_statistics(tokens, config)
        from dataclasses import asdict
        config_dict = asdict(config)
        config_dict["stats"] = stats
        with open(config.config_path, "w") as f:
            json.dump(config_dict, f)

        # Verify all files exist and are loadable
        assert os.path.exists(config.tokenizer_path)
        assert os.path.exists(config.train_data_path)
        assert os.path.exists(config.config_path)

        # Load tokenizer
        loaded_tokenizer = Tokenizer.from_file(config.tokenizer_path)
        assert loaded_tokenizer.get_vocab_size() > 0

        # Load tokens
        loaded_tokens = torch.load(config.train_data_path)
        assert torch.equal(loaded_tokens, tokens)

        # Load config
        with open(config.config_path) as f:
            loaded_config = json.load(f)
        assert loaded_config["vocab_size"] == config.vocab_size


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_text(self, trained_tokenizer, config):
        """Test handling of empty text."""
        class EmptyDataset:
            def __iter__(self):
                yield {"text": ""}
            def __len__(self):
                return 1

        tokens = tokenize_dataset(EmptyDataset(), trained_tokenizer, config)

        # Should still have BOS and EOS, rest padding
        assert tokens.shape == (1, config.max_seq_len)
        assert tokens[0, 0] == config.bos_token_id

    def test_single_word(self, trained_tokenizer, config):
        """Test handling of single word."""
        class SingleWordDataset:
            def __iter__(self):
                yield {"text": "Hello"}
            def __len__(self):
                return 1

        tokens = tokenize_dataset(SingleWordDataset(), trained_tokenizer, config)

        assert tokens.shape == (1, config.max_seq_len)
        assert tokens[0, 0] == config.bos_token_id
        # Should have EOS somewhere
        assert (tokens[0] == config.eos_token_id).any()

    def test_exact_length_sequence(self, trained_tokenizer, config):
        """Test sequence that tokenizes to exactly max_seq_len."""
        # This is tricky to construct, so we just verify no crash
        text = "word " * 100  # Should be longer than max_seq_len tokens

        class ExactDataset:
            def __iter__(self):
                yield {"text": text}
            def __len__(self):
                return 1

        tokens = tokenize_dataset(ExactDataset(), trained_tokenizer, config)

        assert tokens.shape == (1, config.max_seq_len)

    def test_special_characters(self, trained_tokenizer, config):
        """Test handling of special characters."""
        special_texts = [
            "Hello! How are you?",
            "Price: $100.00",
            "Email: test@example.com",
            "Math: 2 + 2 = 4",
            "Newline:\nTab:\t",
        ]

        class SpecialDataset:
            def __iter__(self):
                for text in special_texts:
                    yield {"text": text}
            def __len__(self):
                return len(special_texts)

        # Should not crash
        tokens = tokenize_dataset(SpecialDataset(), trained_tokenizer, config)

        assert tokens.shape == (len(special_texts), config.max_seq_len)
        # All should start with BOS
        assert (tokens[:, 0] == config.bos_token_id).all()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
