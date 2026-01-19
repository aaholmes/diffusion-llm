#!/usr/bin/env python3
"""
Generate captions from images using trained discrete diffusion captioning model.

Usage:
    # From image file
    python generate_caption.py --checkpoint checkpoints_caption_poc/best.pt --image path/to/image.jpg

    # From validation set
    python generate_caption.py --checkpoint checkpoints_caption_poc/best.pt --use_val_set --num_samples 5
"""

import argparse
import json
import os

import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor

from src.core.model import DiffusionTransformer, ModelConfig
from src.core.diffusion import DiscreteDiffusion


def load_captioning_model(checkpoint_path: str, device: str = "cpu"):
    """Load decoder from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load decoder
    decoder_config = checkpoint['decoder_config']
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)

    decoder = DiffusionTransformer(decoder_config)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.to(device)
    decoder.eval()

    print(f"  Decoder: {decoder_config.n_layers} layers, d_model={decoder_config.d_model}")

    return decoder, checkpoint.get('train_config', {})


@torch.no_grad()
def extract_image_features(image: Image.Image, vision_model, image_processor, device: str = "cpu"):
    """Extract CLIP features from image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    outputs = vision_model(pixel_values)
    image_features = outputs.last_hidden_state  # [1, seq_len, d_model]

    return image_features


@torch.no_grad()
def generate_caption(
    decoder,
    diffusion,
    image_features: torch.Tensor,
    max_len: int = 128,
    steps: int = 50,
    temperature: float = 0.8,
    device: str = "cpu",
    mask_token_id: int = 3,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Generate caption from image features."""
    batch_size = image_features.shape[0]
    vocab_size = decoder.vocab_size

    # Move image features to device
    image_features = image_features.to(device)

    # Create attention mask for image features (all valid)
    image_mask = torch.ones(image_features.shape[:2], device=device)

    # Start with all masks
    x = torch.full((batch_size, max_len), mask_token_id, device=device)

    # Iteratively denoise
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for step in range(steps):
        t = timesteps[step].expand(batch_size)
        t_next = timesteps[step + 1].expand(batch_size)

        logits = decoder(
            x, t,
            encoder_output=image_features,
            encoder_attention_mask=image_mask,
        )

        # Sample from logits
        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1)
        sampled = sampled.view(batch_size, max_len)

        # Unmask based on timestep (cosine schedule)
        current_mask_rate = 1 - torch.cos(t * torch.pi / 2)
        next_mask_rate = 1 - torch.cos(t_next * torch.pi / 2)

        keep_mask_prob = (next_mask_rate / current_mask_rate.clamp(min=1e-8)).clamp(max=1.0)

        is_masked = (x == mask_token_id)
        rand = torch.rand(batch_size, max_len, device=device)
        keep_mask = rand < keep_mask_prob.unsqueeze(1)

        unmask = is_masked & ~keep_mask
        x[unmask] = sampled[unmask]

    return x


def main():
    parser = argparse.ArgumentParser(description="Generate captions from images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--use_val_set", action="store_true", help="Generate from validation set")
    parser.add_argument("--val_index", type=int, default=0, help="Start index in validation set")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--data_dir", type=str, default="data_captions_synthetic")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--max_len", type=int, default=None, help="Max caption length (overrides checkpoint config)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    mask_id = tokenizer.token_to_id("<MASK>")

    # Load model
    decoder, train_config = load_captioning_model(args.checkpoint, device)
    max_caption_len = args.max_len or train_config.get('max_caption_len', 128)

    # Create diffusion
    diffusion = DiscreteDiffusion(
        vocab_size=decoder.vocab_size,
        mask_token_id=mask_id,
        pad_token_id=pad_id,
        schedule="cosine",
    )

    # Load CLIP vision encoder if needed
    vision_model = None
    image_processor = None

    if not args.use_val_set:
        if not args.image:
            print("Error: Must specify --image or --use_val_set")
            return

        print(f"Loading CLIP model: {args.clip_model}")
        vision_model = CLIPVisionModel.from_pretrained(args.clip_model)
        vision_model.to(device)
        vision_model.eval()
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

        # Load and process image
        print(f"\nLoading image: {args.image}")
        image = Image.open(args.image)
        image_features = extract_image_features(image, vision_model, image_processor, device)

        print(f"Generating {args.num_samples} captions (steps={args.steps}, temp={args.temperature})...")

        # Generate multiple samples
        image_features_batch = image_features.repeat(args.num_samples, 1, 1)

        output = generate_caption(
            decoder, diffusion, image_features_batch,
            max_len=max_caption_len,
            steps=args.steps,
            temperature=args.temperature,
            device=device,
            mask_token_id=mask_id,
            pad_token_id=pad_id,
        )

        # Decode and print
        print("\n" + "="*70)
        print("Generated Captions:")
        print("="*70)

        for i in range(args.num_samples):
            tokens = output[i].tolist()
            tokens = [t for t in tokens if t not in [pad_id, bos_id, eos_id, mask_id]]
            text = tokenizer.decode(tokens)
            print(f"\n[{i+1}] {text}")

        print("\n" + "="*70)

    else:
        # Use validation set
        print("Loading validation data...")
        val_features = torch.load(os.path.join(args.data_dir, "val_image_features.pt"))
        val_captions = torch.load(os.path.join(args.data_dir, "val_captions.pt"))

        print(f"  Validation examples: {len(val_captions)}")
        print(f"  Starting from index: {args.val_index}")

        print("\n" + "="*70)
        print("Generating Captions from Validation Set:")
        print("="*70)

        for idx in range(args.val_index, min(args.val_index + args.num_samples, len(val_captions))):
            print(f"\n--- Example {idx} ---")

            # Get ground truth
            gt_tokens = val_captions[idx].tolist()
            gt_tokens = [t for t in gt_tokens if t not in [pad_id, bos_id, eos_id, mask_id]]
            gt_text = tokenizer.decode(gt_tokens)
            print(f"Ground truth: {gt_text}")

            # Generate
            image_features = val_features[idx:idx+1].to(device)

            output = generate_caption(
                decoder, diffusion, image_features,
                max_len=max_caption_len,
                steps=args.steps,
                temperature=args.temperature,
                device=device,
                mask_token_id=mask_id,
                pad_token_id=pad_id,
            )

            # Decode
            pred_tokens = output[0].tolist()
            pred_tokens = [t for t in pred_tokens if t not in [pad_id, bos_id, eos_id, mask_id]]
            pred_text = tokenizer.decode(pred_tokens)
            print(f"Generated:    {pred_text}")

        print("\n" + "="*70)


if __name__ == "__main__":
    main()
