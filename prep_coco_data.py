#!/usr/bin/env python3
"""
Prepare COCO Captions dataset for training.

Downloads COCO 2017 captions and images, extracts CLIP features,
and saves in the same format as synthetic data.

Usage:
    # Download and prepare full dataset (requires ~20GB disk space)
    python prep_coco_data.py --num_train 10000 --num_val 1000

    # Quick test with small subset
    python prep_coco_data.py --num_train 100 --num_val 50 --output_dir data_coco_test
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile

import torch
from PIL import Image
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    if dest.exists():
        print(f"  {dest.name} already exists, skipping download")
        return

    print(f"  Downloading {desc}...")

    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        print(f"\r  Progress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()


def download_coco_annotations(data_dir: Path):
    """Download COCO 2017 caption annotations."""
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # COCO 2017 annotations
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = data_dir / "annotations_trainval2017.zip"

    download_file(url, zip_path, "COCO 2017 annotations")

    # Extract if needed
    if not (annotations_dir / "captions_train2017.json").exists():
        print("  Extracting annotations...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)

    return annotations_dir


def load_coco_captions(annotations_file: Path) -> dict:
    """Load COCO captions and return image_id -> captions mapping."""
    print(f"  Loading {annotations_file.name}...")
    with open(annotations_file) as f:
        data = json.load(f)

    # Build image_id -> captions mapping
    captions_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in captions_by_image:
            captions_by_image[img_id] = []
        captions_by_image[img_id].append(ann['caption'])

    # Build image_id -> filename mapping
    id_to_file = {img['id']: img['file_name'] for img in data['images']}

    return captions_by_image, id_to_file


def download_coco_images(
    image_ids: list,
    id_to_file: dict,
    images_dir: Path,
    split: str = "train2017"
):
    """Download specific COCO images."""
    images_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://images.cocodataset.org/{split}"

    downloaded = 0
    skipped = 0

    for img_id in tqdm(image_ids, desc=f"Downloading {split} images"):
        filename = id_to_file[img_id]
        dest = images_dir / filename

        if dest.exists():
            skipped += 1
            continue

        url = f"{base_url}/{filename}"
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
        except Exception as e:
            print(f"\n  Warning: Failed to download {filename}: {e}")

    print(f"  Downloaded {downloaded}, skipped {skipped} existing")


@torch.no_grad()
def extract_clip_features(
    image_paths: list,
    clip_model: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract CLIP features for a list of images."""
    print(f"  Loading CLIP model: {clip_model}")
    vision_model = CLIPVisionModel.from_pretrained(clip_model)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model)
    vision_model.to(device)
    vision_model.eval()

    all_features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP features"):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"\n  Warning: Failed to load {path}: {e}")
                # Use a blank image as fallback
                images.append(Image.new('RGB', (224, 224), color='white'))

        inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        outputs = vision_model(pixel_values)
        features = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def tokenize_captions(
    captions: list,
    tokenizer: Tokenizer,
    max_len: int = 64,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
) -> torch.Tensor:
    """Tokenize captions with padding."""
    all_tokens = []

    for caption in tqdm(captions, desc="Tokenizing captions"):
        # Tokenize
        encoded = tokenizer.encode(caption)
        tokens = encoded.ids

        # Add BOS/EOS
        tokens = [bos_id] + tokens + [eos_id]

        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len-1] + [eos_id]
        else:
            tokens = tokens + [pad_id] * (max_len - len(tokens))

        all_tokens.append(tokens)

    return torch.tensor(all_tokens, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO captions dataset")
    parser.add_argument("--num_train", type=int, default=10000,
                        help="Number of training examples")
    parser.add_argument("--num_val", type=int, default=1000,
                        help="Number of validation examples")
    parser.add_argument("--output_dir", type=str, default="data_coco",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="data_coco_cache",
                        help="Cache directory for downloads")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json",
                        help="Path to tokenizer")
    parser.add_argument("--max_caption_len", type=int, default=64,
                        help="Maximum caption length in tokens")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model for feature extraction")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for CLIP feature extraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for CLIP inference")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COCO Captions Dataset Preparation")
    print("=" * 60)
    print(f"  Train examples: {args.num_train}")
    print(f"  Val examples: {args.num_val}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {args.device}")
    print()

    # Step 1: Download annotations
    print("Step 1: Downloading annotations...")
    annotations_dir = download_coco_annotations(cache_dir)

    # Step 2: Load captions
    print("\nStep 2: Loading captions...")
    train_captions, train_id_to_file = load_coco_captions(
        annotations_dir / "captions_train2017.json"
    )
    val_captions, val_id_to_file = load_coco_captions(
        annotations_dir / "captions_val2017.json"
    )
    print(f"  Train images with captions: {len(train_captions)}")
    print(f"  Val images with captions: {len(val_captions)}")

    # Step 3: Sample subset
    print("\nStep 3: Sampling subset...")
    train_image_ids = random.sample(list(train_captions.keys()),
                                     min(args.num_train, len(train_captions)))
    val_image_ids = random.sample(list(val_captions.keys()),
                                   min(args.num_val, len(val_captions)))
    print(f"  Selected {len(train_image_ids)} train, {len(val_image_ids)} val images")

    # Step 4: Download images
    print("\nStep 4: Downloading images...")
    train_images_dir = cache_dir / "train2017"
    val_images_dir = cache_dir / "val2017"

    download_coco_images(train_image_ids, train_id_to_file, train_images_dir, "train2017")
    download_coco_images(val_image_ids, val_id_to_file, val_images_dir, "val2017")

    # Step 5: Load tokenizer
    print("\nStep 5: Loading tokenizer...")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Step 6: Extract CLIP features and prepare data
    print("\nStep 6: Processing training data...")

    # Get one caption per image for training (randomly select from 5)
    train_image_paths = []
    train_caption_texts = []
    for img_id in train_image_ids:
        filename = train_id_to_file[img_id]
        path = train_images_dir / filename
        if path.exists():
            train_image_paths.append(path)
            # Randomly select one of the 5 captions
            train_caption_texts.append(random.choice(train_captions[img_id]))

    print(f"  Processing {len(train_image_paths)} training images...")
    train_features = extract_clip_features(
        train_image_paths, args.clip_model, args.batch_size, args.device
    )
    train_tokens = tokenize_captions(train_caption_texts, tokenizer, args.max_caption_len)

    print("\nStep 7: Processing validation data...")
    val_image_paths = []
    val_caption_texts = []
    for img_id in val_image_ids:
        filename = val_id_to_file[img_id]
        path = val_images_dir / filename
        if path.exists():
            val_image_paths.append(path)
            val_caption_texts.append(random.choice(val_captions[img_id]))

    print(f"  Processing {len(val_image_paths)} validation images...")
    val_features = extract_clip_features(
        val_image_paths, args.clip_model, args.batch_size, args.device
    )
    val_tokens = tokenize_captions(val_caption_texts, tokenizer, args.max_caption_len)

    # Step 8: Save data
    print("\nStep 8: Saving processed data...")

    torch.save(train_features, output_dir / "train_image_features.pt")
    torch.save(train_tokens, output_dir / "train_captions.pt")
    torch.save(val_features, output_dir / "val_image_features.pt")
    torch.save(val_tokens, output_dir / "val_captions.pt")

    # Save config
    config = {
        "clip_model": args.clip_model,
        "feature_dim": train_features.shape[-1],
        "seq_len": train_features.shape[1],
        "max_caption_len": args.max_caption_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer": args.tokenizer,
        "dataset_type": "coco",
        "num_train": len(train_image_paths),
        "num_val": len(val_image_paths),
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save captions for reference
    with open(output_dir / "train_captions.json", 'w') as f:
        json.dump(train_caption_texts, f)
    with open(output_dir / "val_captions.json", 'w') as f:
        json.dump(val_caption_texts, f)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Train features: {train_features.shape}")
    print(f"  Train captions: {train_tokens.shape}")
    print(f"  Val features: {val_features.shape}")
    print(f"  Val captions: {val_tokens.shape}")
    print(f"\nOutput saved to: {args.output_dir}")
    print(f"\nTo train:")
    print(f"  python train_captioning.py --data_dir {args.output_dir} --max_steps 10000")


if __name__ == "__main__":
    main()
