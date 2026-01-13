#!/usr/bin/env python3
"""
Prepare image captioning data using COCO dataset.

This script:
1. Downloads COCO captions dataset (via HuggingFace datasets)
2. Extracts visual features using CLIP vision encoder
3. Tokenizes captions
4. Saves as tensors for training

Usage:
    python prep_caption_data.py --output_dir data_captions --max_train 50000 --max_val 5000
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO caption data")
    parser.add_argument("--output_dir", type=str, default="data_captions")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--max_train", type=int, default=50000, help="Max training examples")
    parser.add_argument("--max_val", type=int, default=5000, help="Max validation examples")
    parser.add_argument("--max_caption_len", type=int, default=128, help="Max caption length")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP vision encoder and processor
    print(f"\nLoading CLIP model: {args.clip_model}")
    vision_model = CLIPVisionModel.from_pretrained(args.clip_model)
    vision_model.to(device)
    vision_model.eval()

    image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = vision_model(dummy_input)
        feature_dim = dummy_output.last_hidden_state.shape[-1]
        seq_len = dummy_output.last_hidden_state.shape[1]

    print(f"  Feature dimension: {feature_dim}")
    print(f"  Sequence length: {seq_len}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")

    print(f"  PAD={pad_id}, BOS={bos_id}, EOS={eos_id}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Load dataset - using Flickr30k which is smaller and easier for POC
    print("\nLoading Flickr30k captions dataset...")
    print("  This may take a few minutes on first run (downloading images)...")

    # Using HuggingFace Flickr30k dataset
    dataset = load_dataset("nlphuji/flickr30k")

    print(f"  Train examples: {len(dataset['test'])}")  # Flickr30k only has test split

    # Split into train/val ourselves
    test_split = dataset['test']
    total = len(test_split)
    train_size = int(0.9 * total)  # 90% train, 10% val

    print(f"  Splitting into train ({train_size}) and validation ({total - train_size})")

    # Get test split and divide into train/val
    all_data = dataset['test']
    total = len(all_data)
    train_size = int(0.9 * total)

    # Process splits
    for split_name, start_idx, end_idx, max_examples in [
        ("train", 0, train_size, args.max_train),
        ("val", train_size, total, args.max_val)
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split (max {max_examples} examples)")
        print(f"{'='*60}")

        # Limit examples
        available = end_idx - start_idx
        num_examples = min(available, max_examples)

        image_features_list = []
        caption_tokens_list = []

        processed = 0
        skipped = 0

        for i in tqdm(range(num_examples), desc=f"Processing {split_name}"):
            try:
                idx = start_idx + i
                example = all_data[idx]

                # Get image
                image = example['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Get caption (Flickr30k has multiple captions, take first one)
                captions = example['caption']
                if isinstance(captions, list):
                    caption_text = captions[0] if len(captions) > 0 else ""
                else:
                    caption_text = captions

                if not caption_text:
                    skipped += 1
                    continue

                # Extract image features
                inputs = image_processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device)

                with torch.no_grad():
                    outputs = vision_model(pixel_values)
                    # Get patch features (not pooled): [1, seq_len, feature_dim]
                    image_features = outputs.last_hidden_state.squeeze(0).cpu()

                # Tokenize caption
                encoded = tokenizer.encode(caption_text)
                caption_ids = [bos_id] + encoded.ids + [eos_id]

                # Check length
                if len(caption_ids) > args.max_caption_len:
                    skipped += 1
                    continue

                # Pad caption
                caption_ids = caption_ids + [pad_id] * (args.max_caption_len - len(caption_ids))

                image_features_list.append(image_features)
                caption_tokens_list.append(caption_ids)
                processed += 1

            except Exception as e:
                print(f"\nError processing example {idx}: {e}")
                skipped += 1
                continue

        print(f"\nProcessed: {processed}, Skipped: {skipped}")

        # Convert to tensors
        print("Converting to tensors...")
        image_features_tensor = torch.stack(image_features_list)
        caption_tokens_tensor = torch.tensor(caption_tokens_list, dtype=torch.long)

        print(f"  Image features: {image_features_tensor.shape}")
        print(f"  Caption tokens: {caption_tokens_tensor.shape}")

        # Save
        features_path = os.path.join(args.output_dir, f"{split_name}_image_features.pt")
        captions_path = os.path.join(args.output_dir, f"{split_name}_captions.pt")

        print(f"Saving to {args.output_dir}...")
        torch.save(image_features_tensor, features_path)
        torch.save(caption_tokens_tensor, captions_path)

        print(f"  ✓ {features_path}")
        print(f"  ✓ {captions_path}")

    # Save config
    config = {
        "clip_model": args.clip_model,
        "feature_dim": feature_dim,
        "seq_len": seq_len,
        "max_caption_len": args.max_caption_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer": args.tokenizer,
    }

    config_path = os.path.join(args.output_dir, "config.json")
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Saved config: {config_path}")
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
