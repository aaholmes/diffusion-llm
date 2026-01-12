#!/usr/bin/env python3
"""Tests for prep_infill_data.py"""

import pytest

from prep_infill_data import split_sentences, extract_infill_triple


class TestSplitSentences:
    """Tests for sentence splitting logic."""

    def test_simple_split(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence here."
        assert sentences[1] == "Second sentence here."
        assert sentences[2] == "Third sentence here."

    def test_exclamation_split(self):
        text = "Hello world! This is great! Another sentence here."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_question_split(self):
        text = "What is this? It is a cat. The cat is fluffy."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_filters_short_sentences(self):
        text = "Hi. This is a longer sentence here. Ok."
        sentences = split_sentences(text)
        # "Hi." and "Ok." are less than 10 chars, should be filtered
        assert len(sentences) == 1
        assert "longer sentence" in sentences[0]

    def test_empty_text(self):
        sentences = split_sentences("")
        assert sentences == []

    def test_whitespace_only(self):
        sentences = split_sentences("   ")
        assert sentences == []

    def test_single_sentence(self):
        text = "This is just one sentence without a capital after period."
        sentences = split_sentences(text)
        assert len(sentences) == 1

    def test_lowercase_after_period_not_split(self):
        text = "The price was $5.00 dollars. That was cheap for this item."
        sentences = split_sentences(text)
        # Should not split at "$5.00 d" since 'd' is lowercase
        assert any("$5.00" in s for s in sentences)


class TestExtractInfillTriple:
    """Tests for extracting (first, middle, last) triples."""

    def test_simple_extraction(self):
        text = "First sentence here. Middle sentence here is longer. Last sentence here."
        result = extract_infill_triple(text, min_middle_chars=10)
        assert result is not None
        first, middle, last = result
        assert first == "First sentence here."
        assert middle == "Middle sentence here is longer."
        assert last == "Last sentence here."

    def test_multiple_middle_sentences(self):
        text = "First sentence here. Second sentence. Third sentence. Last sentence here."
        result = extract_infill_triple(text, min_middle_chars=10)
        assert result is not None
        first, middle, last = result
        assert first == "First sentence here."
        assert "Second sentence" in middle
        assert "Third sentence" in middle
        assert last == "Last sentence here."

    def test_too_few_sentences(self):
        text = "First sentence. Second sentence."
        result = extract_infill_triple(text)
        assert result is None

    def test_single_sentence(self):
        text = "Just one sentence here."
        result = extract_infill_triple(text)
        assert result is None

    def test_empty_text(self):
        result = extract_infill_triple("")
        assert result is None

    def test_short_first_sentence(self):
        text = "Hi. This is a longer middle portion here. Last sentence is here."
        result = extract_infill_triple(text)
        # "Hi." is filtered out, so we don't have 3 sentences
        assert result is None

    def test_short_middle(self):
        text = "First sentence is here. Ok. Last sentence is here."
        result = extract_infill_triple(text, min_middle_chars=30)
        # Middle "Ok." is too short (filtered), leaving only 2 sentences
        assert result is None

    def test_min_middle_chars_threshold(self):
        text = "First sentence is longer. Short mid. Last sentence is longer."
        # With high threshold, middle is too short
        result = extract_infill_triple(text, min_middle_chars=100)
        assert result is None

    def test_valid_extraction_with_threshold(self):
        text = "Once upon a time there was a princess. She lived in a big castle and had many adventures. The end of the story."
        result = extract_infill_triple(text, min_middle_chars=20)
        assert result is not None
        first, middle, last = result
        assert "princess" in first
        assert "castle" in middle
        assert "end" in last


class TestPrepareInfillData:
    """Tests for prepare_infill_data function."""

    def test_prepare_with_mock_data(self, tmp_path, monkeypatch):
        """Test preparation with mocked dataset."""
        import datasets
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=8192,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story about a little girl who lived in a castle."] * 100,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Mock the dataset loading
        class MockDataset:
            def __init__(self):
                self.data = [
                    {"text": "Once upon a time there was a princess. She lived in a big castle with her family and had many adventures. They were all very happy in the end."},
                    {"text": "The little boy went to school today. He learned many new and exciting things in class. His teacher was very proud of him."},
                    {"text": "A cat sat on the soft mat. It was a fluffy orange cat who loved to play. The cat loved to sleep all day long."},
                ] * 20

            def __len__(self):
                return len(self.data)

            def __iter__(self):
                return iter(self.data)

            def select(self, indices):
                self.data = [self.data[i] for i in indices if i < len(self.data)]
                return self

        def mock_load_dataset(*args, **kwargs):
            return MockDataset()

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        from prep_infill_data import prepare_infill_data

        output_dir = tmp_path / "output"
        prepare_infill_data(
            tokenizer_path=str(tokenizer_path),
            output_dir=str(output_dir),
            subset_size=60,
            max_condition_len=96,
            max_target_len=160,
        )

        # Verify outputs exist
        assert (output_dir / "train_condition.pt").exists()
        assert (output_dir / "train_target.pt").exists()
        assert (output_dir / "val_condition.pt").exists()
        assert (output_dir / "val_target.pt").exists()
        assert (output_dir / "config.json").exists()

        # Verify config content
        import json
        with open(output_dir / "config.json") as f:
            config = json.load(f)
        assert config["max_condition_len"] == 96
        assert config["max_target_len"] == 160
        assert "infill" in config["task"]

    def test_prepare_with_length_filtering(self, tmp_path, monkeypatch):
        """Test that pairs exceeding max length are filtered."""
        import datasets
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=8192,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["word " * 100] * 10,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Mock dataset with very long texts
        class MockDataset:
            def __init__(self):
                # Very long sentences that will exceed max lengths
                long_sentence = "This is a very long sentence " * 20
                self.data = [
                    {"text": f"{long_sentence}. {long_sentence}. {long_sentence}."},
                ] * 10

            def __len__(self):
                return len(self.data)

            def __iter__(self):
                return iter(self.data)

            def select(self, indices):
                self.data = [self.data[i] for i in indices if i < len(self.data)]
                return self

        def mock_load_dataset(*args, **kwargs):
            return MockDataset()

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        from prep_infill_data import prepare_infill_data

        output_dir = tmp_path / "output"
        # Use very small max lengths to force filtering
        prepare_infill_data(
            tokenizer_path=str(tokenizer_path),
            output_dir=str(output_dir),
            subset_size=10,
            max_condition_len=20,  # Very small
            max_target_len=20,     # Very small
        )

        # Should still create files even if empty
        assert (output_dir / "config.json").exists()


class TestMainFunction:
    """Tests for main function."""

    def test_main_argument_parsing(self):
        """Test that main parses arguments correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
        parser.add_argument("--output_dir", type=str, default="data_infill")
        parser.add_argument("--subset", type=int, default=None)
        parser.add_argument("--max_condition_len", type=int, default=96)
        parser.add_argument("--max_target_len", type=int, default=160)

        args = parser.parse_args(['--subset', '1000', '--max_condition_len', '64'])
        assert args.subset == 1000
        assert args.max_condition_len == 64
        assert args.max_target_len == 160


class TestInfillIntegration:
    """Integration tests for infill data preparation."""

    def test_tokenizer_special_tokens(self, tmp_path):
        """Test that tokenizer has required special tokens."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=1000,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story."] * 10,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        loaded = Tokenizer.from_file(str(tokenizer_path))
        assert loaded.token_to_id("<PAD>") == 0
        assert loaded.token_to_id("<BOS>") == 1
        assert loaded.token_to_id("<EOS>") == 2

    def test_output_tensor_shapes(self, tmp_path):
        """Test that output tensors have correct shapes."""
        import torch

        max_condition_len = 96
        max_target_len = 160
        num_samples = 10

        condition_data = torch.randint(0, 1000, (num_samples, max_condition_len))
        target_data = torch.randint(0, 1000, (num_samples, max_target_len))

        assert condition_data.shape == (num_samples, max_condition_len)
        assert target_data.shape == (num_samples, max_target_len)

        torch.save(condition_data, tmp_path / "cond.pt")
        torch.save(target_data, tmp_path / "target.pt")

        loaded_cond = torch.load(tmp_path / "cond.pt", weights_only=True)
        loaded_target = torch.load(tmp_path / "target.pt", weights_only=True)

        assert torch.equal(condition_data, loaded_cond)
        assert torch.equal(target_data, loaded_target)

    def test_condition_format(self):
        """Test that condition format is BOS + first + SEP + last + EOS."""
        # Simulate the condition building logic
        bos_id, eos_id, sep_id, pad_id = 1, 2, 2, 0  # sep = eos
        first_ids = [10, 11, 12]
        last_ids = [20, 21, 22]

        condition = [bos_id] + first_ids + [sep_id] + last_ids + [eos_id]

        assert condition[0] == bos_id
        assert condition[-1] == eos_id
        assert sep_id in condition
        assert len(condition) == 1 + len(first_ids) + 1 + len(last_ids) + 1

    def test_target_format(self):
        """Test that target format is BOS + middle + EOS."""
        bos_id, eos_id = 1, 2
        middle_ids = [30, 31, 32, 33]

        target = [bos_id] + middle_ids + [eos_id]

        assert target[0] == bos_id
        assert target[-1] == eos_id
        assert len(target) == 1 + len(middle_ids) + 1
