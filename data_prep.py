#!/usr/bin/env python3
"""
Data preparation pipeline for diffusion language model.

Downloads TinyStories dataset, trains a BPE tokenizer, and tokenizes the dataset.

Usage:
    python data_prep.py                    # Run full pipeline with defaults
    python data_prep.py --subset_size 10000  # Use smaller subset for testing
    python data_prep.py --vocab_size 16384   # Larger vocabulary
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tqdm import tqdm


@dataclass
class DataConfig:
    """Configuration for data preparation."""
    # Dataset
    dataset_name: str = "roneneldan/TinyStories"
    subset_size: int = 100_000  # Number of stories to use (None for full dataset)
    val_split: float = 0.05  # Fraction for validation

    # Tokenizer
    vocab_size: int = 8192
    min_frequency: int = 2  # Minimum token frequency

    # Tokenization
    max_seq_len: int = 256

    # Paths
    data_dir: str = "data"
    tokenizer_path: str = "data/tokenizer.json"
    train_data_path: str = "data/train_tokens.pt"
    val_data_path: str = "data/val_tokens.pt"
    config_path: str = "data/config.json"

    # Special tokens (order matters - indices are assigned sequentially)
    special_tokens: tuple = ("<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>")

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def mask_token_id(self) -> int:
        return 3

    @property
    def unk_token_id(self) -> int:
        return 4


def load_stories(config: DataConfig) -> tuple:
    """
    Load TinyStories dataset from HuggingFace.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print(f"Loading dataset: {config.dataset_name}")

    # Load the full training split
    dataset = load_dataset(config.dataset_name, split="train")
    print(f"Full dataset size: {len(dataset):,} stories")

    # Take subset if specified
    if config.subset_size is not None and config.subset_size < len(dataset):
        dataset = dataset.select(range(config.subset_size))
        print(f"Using subset: {len(dataset):,} stories")

    # Split into train/val
    split = dataset.train_test_split(test_size=config.val_split, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Train: {len(train_dataset):,} stories")
    print(f"Val: {len(val_dataset):,} stories")

    return train_dataset, val_dataset


def train_tokenizer(dataset, config: DataConfig) -> Tokenizer:
    """
    Train a BPE tokenizer on the dataset.

    Args:
        dataset: HuggingFace dataset with "text" field
        config: Data configuration

    Returns:
        Trained tokenizer
    """
    print(f"\nTraining BPE tokenizer (vocab_size={config.vocab_size})...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    # Pre-tokenizer: split on whitespace and punctuation, handle bytes
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder to convert back from byte-level
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        show_progress=True,
    )

    # Batch iterator for training
    def batch_iterator(batch_size: int = 1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]["text"]
            yield batch

    # Train the tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Add post-processor to wrap sequences with BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<BOS> $A <EOS>",
        special_tokens=[
            ("<BOS>", config.bos_token_id),
            ("<EOS>", config.eos_token_id),
        ],
    )

    # Verify special tokens
    print("\nSpecial tokens:")
    for token in config.special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")

    # Test encoding
    test_text = "Once upon a time, there was a little girl."
    encoding = tokenizer.encode(test_text)
    print(f"\nTest encoding:")
    print(f"  Input: {test_text}")
    print(f"  Tokens: {encoding.tokens[:20]}...")
    print(f"  IDs: {encoding.ids[:20]}...")

    # Save tokenizer
    os.makedirs(os.path.dirname(config.tokenizer_path), exist_ok=True)
    tokenizer.save(config.tokenizer_path)
    print(f"\nTokenizer saved to: {config.tokenizer_path}")

    return tokenizer


def tokenize_dataset(
    dataset,
    tokenizer: Tokenizer,
    config: DataConfig,
    desc: str = "Tokenizing"
) -> torch.Tensor:
    """
    Tokenize a dataset and return as a tensor.

    Args:
        dataset: HuggingFace dataset with "text" field
        tokenizer: Trained tokenizer
        config: Data configuration
        desc: Progress bar description

    Returns:
        Tensor of shape [num_samples, max_seq_len]
    """
    all_tokens = []

    # Tokenize with progress bar
    for example in tqdm(dataset, desc=desc):
        # Encode text (BOS and EOS added by post-processor)
        encoding = tokenizer.encode(example["text"])
        tokens = encoding.ids

        # Truncate if too long
        if len(tokens) > config.max_seq_len:
            tokens = tokens[:config.max_seq_len - 1] + [config.eos_token_id]

        # Pad if too short
        if len(tokens) < config.max_seq_len:
            padding = [config.pad_token_id] * (config.max_seq_len - len(tokens))
            tokens = tokens + padding

        all_tokens.append(tokens)

    # Convert to tensor
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)

    return tokens_tensor


def compute_statistics(tokens: torch.Tensor, config: DataConfig) -> dict:
    """Compute dataset statistics."""
    # Non-padding tokens per sequence
    non_pad = (tokens != config.pad_token_id).sum(dim=1).float()

    stats = {
        "num_sequences": len(tokens),
        "seq_length": config.max_seq_len,
        "total_tokens": int(tokens.numel()),
        "non_pad_tokens": int((tokens != config.pad_token_id).sum()),
        "avg_length": float(non_pad.mean()),
        "min_length": int(non_pad.min()),
        "max_length": int(non_pad.max()),
        "pad_fraction": float((tokens == config.pad_token_id).float().mean()),
    }

    return stats


def main(config: DataConfig):
    """Run the full data preparation pipeline."""
    print("=" * 60)
    print("Diffusion LM Data Preparation")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in asdict(config).items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")
    print()

    # Create output directory
    os.makedirs(config.data_dir, exist_ok=True)

    # Step 1: Load dataset
    train_dataset, val_dataset = load_stories(config)

    # Step 2: Train tokenizer (on training data only)
    tokenizer = train_tokenizer(train_dataset, config)

    # Step 3: Tokenize datasets
    print("\nTokenizing training data...")
    train_tokens = tokenize_dataset(train_dataset, tokenizer, config, desc="Train")

    print("\nTokenizing validation data...")
    val_tokens = tokenize_dataset(val_dataset, tokenizer, config, desc="Val")

    # Step 4: Compute and display statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    train_stats = compute_statistics(train_tokens, config)
    print("\nTraining set:")
    for key, value in train_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")

    val_stats = compute_statistics(val_tokens, config)
    print("\nValidation set:")
    for key, value in val_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}")

    # Step 5: Save tokenized data
    print("\n" + "=" * 60)
    print("Saving data...")
    print("=" * 60)

    torch.save(train_tokens, config.train_data_path)
    print(f"Train tokens saved to: {config.train_data_path}")
    print(f"  Shape: {train_tokens.shape}")

    torch.save(val_tokens, config.val_data_path)
    print(f"Val tokens saved to: {config.val_data_path}")
    print(f"  Shape: {val_tokens.shape}")

    # Save config for later use
    config_dict = asdict(config)
    config_dict["train_stats"] = train_stats
    config_dict["val_stats"] = val_stats

    with open(config.config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to: {config.config_path}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)

    return train_tokens, val_tokens, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for diffusion LM training")

    parser.add_argument("--subset_size", type=int, default=100_000,
                        help="Number of stories to use (default: 100000)")
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="Tokenizer vocabulary size (default: 8192)")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Output directory for data (default: data)")

    args = parser.parse_args()

    # Create config from args
    config = DataConfig(
        subset_size=args.subset_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        data_dir=args.data_dir,
        tokenizer_path=f"{args.data_dir}/tokenizer.json",
        train_data_path=f"{args.data_dir}/train_tokens.pt",
        val_data_path=f"{args.data_dir}/val_tokens.pt",
        config_path=f"{args.data_dir}/config.json",
    )

    main(config)
