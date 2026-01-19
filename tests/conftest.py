#!/usr/bin/env python3
"""
Shared pytest fixtures for all tests.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_tokenizer():
    """Create a minimal mock tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=1000,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", "<IGNORE>"]
    )

    # Train on minimal data
    tokenizer.train_from_iterator(
        ["the quick brown fox", "jumps over the lazy dog"] * 100,
        trainer
    )

    return tokenizer


@pytest.fixture
def saved_tokenizer(temp_dir, mock_tokenizer):
    """Save tokenizer to temp directory and return path."""
    path = os.path.join(temp_dir, "tokenizer.json")
    mock_tokenizer.save(path)
    return path


@pytest.fixture
def mock_train_data(temp_dir):
    """Create mock training and validation data."""
    # Create small token tensors (skip special tokens 0-5)
    train_tokens = torch.randint(6, 500, (100, 32))
    val_tokens = torch.randint(6, 500, (20, 32))

    train_path = os.path.join(temp_dir, "train_tokens.pt")
    val_path = os.path.join(temp_dir, "val_tokens.pt")

    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
