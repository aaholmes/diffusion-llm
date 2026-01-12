#!/usr/bin/env python3
"""
Prepare data for infilling task: given first and last sentence, predict the middle.

This creates a more constrained conditioning task than first→rest, because
the output is bounded on both ends.

Usage:
    python prep_infill_data.py
    python prep_infill_data.py --subset 10000  # Quick test
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitter: split on .!? followed by space and capital
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text.strip())
    # Filter empty and very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def extract_infill_triple(text: str, min_middle_chars: int = 30) -> tuple[str, str, str] | None:
    """
    Extract (first_sentence, middle, last_sentence) from text.

    Returns None if text doesn't have enough sentences or middle is too short.
    """
    sentences = split_sentences(text)

    if len(sentences) < 3:
        return None

    first = sentences[0]
    last = sentences[-1]
    middle = " ".join(sentences[1:-1])

    # Validate lengths
    if len(first) < 15 or len(last) < 15:
        return None
    if len(middle) < min_middle_chars:
        return None

    return first, middle, last


def prepare_infill_data(
    tokenizer_path: str = "data_full/tokenizer.json",
    output_dir: str = "data_infill",
    subset_size: int | None = None,
    max_condition_len: int = 96,  # first + last combined
    max_target_len: int = 160,    # middle portion
    val_split: float = 0.02,
):
    """Prepare infilling data from TinyStories."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Special tokens
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    # We'll use a separator between first and last sentence
    # Using EOS as separator since it's available
    sep_id = eos_id

    print(f"Special tokens: PAD={pad_id}, BOS={bos_id}, EOS={eos_id}, SEP={sep_id}")

    # Load dataset
    print("\nLoading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    if subset_size:
        print(f"Using subset of {subset_size} stories")
        dataset = dataset.select(range(min(subset_size, len(dataset))))

    # Extract triples
    print("\nExtracting (first, middle, last) triples...")
    triples = []

    for item in tqdm(dataset, desc="Processing"):
        text = item["text"]
        triple = extract_infill_triple(text)
        if triple:
            triples.append(triple)

    print(f"Extracted {len(triples):,} valid triples from {len(dataset):,} stories")
    print(f"Extraction rate: {100 * len(triples) / len(dataset):.1f}%")

    # Tokenize
    print("\nTokenizing...")
    condition_tokens = []  # first + SEP + last
    target_tokens = []     # middle

    for first, middle, last in tqdm(triples, desc="Tokenizing"):
        # Tokenize each part
        first_enc = tokenizer.encode(first)
        middle_enc = tokenizer.encode(middle)
        last_enc = tokenizer.encode(last)

        # Build condition: BOS + first + SEP + last + EOS
        condition = [bos_id] + first_enc.ids + [sep_id] + last_enc.ids + [eos_id]

        # Build target: BOS + middle + EOS
        target = [bos_id] + middle_enc.ids + [eos_id]

        # Skip if too long
        if len(condition) > max_condition_len or len(target) > max_target_len:
            continue

        # Pad
        condition = condition + [pad_id] * (max_condition_len - len(condition))
        target = target + [pad_id] * (max_target_len - len(target))

        condition_tokens.append(condition)
        target_tokens.append(target)

    print(f"After length filtering: {len(condition_tokens):,} pairs")

    # Convert to tensors
    condition_tensor = torch.tensor(condition_tokens, dtype=torch.long)
    target_tensor = torch.tensor(target_tokens, dtype=torch.long)

    # Split train/val
    n_total = len(condition_tensor)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_condition = condition_tensor[train_indices]
    train_target = target_tensor[train_indices]
    val_condition = condition_tensor[val_indices]
    val_target = target_tensor[val_indices]

    # Save
    print(f"\nSaving to {output_dir}/")
    torch.save(train_condition, output_dir / "train_condition.pt")
    torch.save(train_target, output_dir / "train_target.pt")
    torch.save(val_condition, output_dir / "val_condition.pt")
    torch.save(val_target, output_dir / "val_target.pt")

    # Save config
    config = {
        "num_train_pairs": len(train_condition),
        "num_val_pairs": len(val_condition),
        "max_condition_len": max_condition_len,
        "max_target_len": max_target_len,
        "tokenizer_path": tokenizer_path,
        "task": "infill (first + last -> middle)",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDataset prepared:")
    print(f"  Train: {len(train_condition):,} pairs")
    print(f"  Val: {len(val_condition):,} pairs")
    print(f"  Condition shape: {train_condition.shape}")
    print(f"  Target shape: {train_target.shape}")

    # Show example
    print("\n" + "=" * 60)
    print("Example:")
    print("=" * 60)

    first, middle, last = triples[0]
    print(f"First: {first}")
    print(f"Last: {last}")
    print(f"Middle: {middle[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare infilling data")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="data_infill")
    parser.add_argument("--subset", type=int, default=None, help="Use subset for testing")
    parser.add_argument("--max_condition_len", type=int, default=96)
    parser.add_argument("--max_target_len", type=int, default=160)

    args = parser.parse_args()

    prepare_infill_data(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        subset_size=args.subset,
        max_condition_len=args.max_condition_len,
        max_target_len=args.max_target_len,
    )


if __name__ == "__main__":
    main()
