#!/usr/bin/env python3
"""
Prepare paired data for conditional training.

Extracts (first sentence, rest of story) pairs from TinyStories.
The model will learn to generate story continuations given a first sentence.

Usage:
    python prep_conditional_data.py                     # From full dataset
    python prep_conditional_data.py --subset 10000      # Quick test
    python prep_conditional_data.py --data_dir data     # From small dataset
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import tqdm


def split_first_sentence(text: str) -> tuple:
    """
    Split text into (first_sentence, rest).

    Returns (None, None) if can't extract a clean split.
    """
    # Clean up text
    text = text.strip()
    if not text:
        return None, None

    # Find first sentence boundary
    # Look for .!? followed by space and capital letter
    pattern = r'([.!?])\s+([A-Z])'
    match = re.search(pattern, text)

    if not match:
        return None, None

    # Split at the boundary
    split_pos = match.start() + 1  # Include the punctuation
    first = text[:split_pos].strip()
    rest = text[split_pos:].strip()

    # Validate
    if len(first) < 10 or len(rest) < 20:
        return None, None

    return first, rest


def prepare_conditional_pairs(
    tokenizer_path: str,
    data_dir: str = "data_full",
    output_dir: str = "data_conditional",
    subset_size: int = None,
    max_encoder_len: int = 64,
    max_decoder_len: int = 192,
):
    """
    Prepare paired (condition, target) data for conditional training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    # Load raw stories
    print("Loading TinyStories dataset...")
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    if subset_size:
        dataset = dataset.select(range(min(subset_size, len(dataset))))

    print(f"Processing {len(dataset)} stories...")

    # Extract pairs
    encoder_inputs = []  # First sentences
    decoder_targets = []  # Rest of stories

    skipped = 0
    for item in tqdm(dataset, desc="Extracting pairs"):
        text = item["text"]
        first, rest = split_first_sentence(text)

        if first is None:
            skipped += 1
            continue

        # Tokenize
        first_enc = tokenizer.encode(first)
        rest_enc = tokenizer.encode(rest)

        first_ids = [bos_id] + first_enc.ids + [eos_id]
        rest_ids = [bos_id] + rest_enc.ids + [eos_id]

        # Check lengths
        if len(first_ids) > max_encoder_len or len(rest_ids) > max_decoder_len:
            skipped += 1
            continue

        # Pad
        first_padded = first_ids + [pad_id] * (max_encoder_len - len(first_ids))
        rest_padded = rest_ids + [pad_id] * (max_decoder_len - len(rest_ids))

        encoder_inputs.append(first_padded)
        decoder_targets.append(rest_padded)

    print(f"\nExtracted {len(encoder_inputs)} pairs ({skipped} skipped)")

    # Convert to tensors
    encoder_tensor = torch.tensor(encoder_inputs, dtype=torch.long)
    decoder_tensor = torch.tensor(decoder_targets, dtype=torch.long)

    # Split train/val
    n = len(encoder_tensor)
    val_size = int(n * 0.02)
    indices = torch.randperm(n)

    train_enc = encoder_tensor[indices[val_size:]]
    train_dec = decoder_tensor[indices[val_size:]]
    val_enc = encoder_tensor[indices[:val_size]]
    val_dec = decoder_tensor[indices[:val_size]]

    # Save
    torch.save(train_enc, f"{output_dir}/train_encoder.pt")
    torch.save(train_dec, f"{output_dir}/train_decoder.pt")
    torch.save(val_enc, f"{output_dir}/val_encoder.pt")
    torch.save(val_dec, f"{output_dir}/val_decoder.pt")

    # Save config
    config = {
        "num_train_pairs": len(train_enc),
        "num_val_pairs": len(val_enc),
        "max_encoder_len": max_encoder_len,
        "max_decoder_len": max_decoder_len,
        "tokenizer_path": tokenizer_path,
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {output_dir}/:")
    print(f"  train_encoder.pt: {train_enc.shape}")
    print(f"  train_decoder.pt: {train_dec.shape}")
    print(f"  val_encoder.pt: {val_enc.shape}")
    print(f"  val_decoder.pt: {val_dec.shape}")

    # Show examples
    print("\n" + "="*60)
    print("Example pairs:")
    print("="*60)
    for i in range(3):
        enc_tokens = train_enc[i].tolist()
        dec_tokens = train_dec[i].tolist()

        # Remove padding and special tokens for display
        enc_tokens = [t for t in enc_tokens if t not in [pad_id, bos_id, eos_id]]
        dec_tokens = [t for t in dec_tokens if t not in [pad_id, bos_id, eos_id]]

        first = tokenizer.decode(enc_tokens)
        rest = tokenizer.decode(dec_tokens)

        print(f"\nPair {i+1}:")
        print(f"  First: {first[:100]}...")
        print(f"  Rest: {rest[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare conditional training data")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="data_conditional")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use subset for quick testing")
    parser.add_argument("--max_encoder_len", type=int, default=64)
    parser.add_argument("--max_decoder_len", type=int, default=192)

    args = parser.parse_args()

    prepare_conditional_pairs(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        subset_size=args.subset,
        max_encoder_len=args.max_encoder_len,
        max_decoder_len=args.max_decoder_len,
    )


if __name__ == "__main__":
    main()
