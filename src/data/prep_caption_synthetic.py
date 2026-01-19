#!/usr/bin/env python3
"""
Create synthetic image captioning data for testing.

This generates random colored images with simple captions like:
- "A red square"
- "A blue circle"
- "A green triangle"

This is just for POC to verify the architecture works!
Then we can swap in real COCO/Flickr data.

Usage:
    python prep_caption_synthetic.py --output_dir data_captions_synthetic
"""

import argparse
import os
import random

import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image, ImageDraw
import numpy as np


COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
SHAPES = ["square", "circle", "triangle", "rectangle"]
SIZES = ["small", "large"]

COLOR_RGB = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
}


def generate_image(color, shape, size):
    """Generate a simple colored shape image."""
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)

    rgb = COLOR_RGB[color]

    # Determine shape size
    if size == "small":
        margin = 80
    else:
        margin = 40

    bbox = [margin, margin, 224-margin, 224-margin]

    if shape == "square" or shape == "rectangle":
        draw.rectangle(bbox, fill=rgb)
    elif shape == "circle":
        draw.ellipse(bbox, fill=rgb)
    elif shape == "triangle":
        points = [(112, margin), (margin, 224-margin), (224-margin, 224-margin)]
        draw.polygon(points, fill=rgb)

    return img


def generate_caption(color, shape, size):
    """Generate a simple caption."""
    templates = [
        f"A {size} {color} {shape}",
        f"There is a {size} {color} {shape}",
        f"This is a {size} {color} {shape}",
        f"A {size} {color} {shape} on a white background",
    ]
    return random.choice(templates)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic caption data")
    parser.add_argument("--output_dir", type=str, default="data_captions_synthetic")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training examples")
    parser.add_argument("--num_val", type=int, default=500, help="Number of validation examples")
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

    # Generate data
    for split_name, num_examples in [("train", args.num_train), ("val", args.num_val)]:
        print(f"\n{'='*60}")
        print(f"Generating {split_name} split ({num_examples} examples)")
        print(f"{'='*60}")

        image_features_list = []
        caption_tokens_list = []

        for idx in tqdm(range(num_examples), desc=f"Generating {split_name}"):
            # Sample random attributes
            color = random.choice(COLORS)
            shape = random.choice(SHAPES)
            size = random.choice(SIZES)

            # Generate image
            image = generate_image(color, shape, size)

            # Generate caption
            caption_text = generate_caption(color, shape, size)

            # Extract image features
            inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)

            with torch.no_grad():
                outputs = vision_model(pixel_values)
                image_features = outputs.last_hidden_state.squeeze(0).cpu()

            # Tokenize caption (tokenizer already adds BOS/EOS)
            encoded = tokenizer.encode(caption_text)
            caption_ids = encoded.ids  # Don't add BOS/EOS - tokenizer handles it

            # Pad caption
            if len(caption_ids) > args.max_caption_len:
                caption_ids = caption_ids[:args.max_caption_len]
            caption_ids = caption_ids + [pad_id] * (args.max_caption_len - len(caption_ids))

            image_features_list.append(image_features)
            caption_tokens_list.append(caption_ids)

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
        "dataset_type": "synthetic",
    }

    config_path = os.path.join(args.output_dir, "config.json")
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Saved config: {config_path}")
    print("\n" + "="*60)
    print("Synthetic data generation complete!")
    print("="*60)
    print("\nExample captions:")
    for i in range(5):
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
        size = random.choice(SIZES)
        print(f"  - {generate_caption(color, shape, size)}")


if __name__ == "__main__":
    main()
