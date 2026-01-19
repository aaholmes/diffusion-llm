#!/usr/bin/env python3
"""
Compare autoregressive and diffusion models side-by-side.
"""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from src.models.model_autoregressive import AutoregressiveCaptioner, AutoregressiveConfig
from src.core.model import DiffusionTransformer, ModelConfig
from src.core.diffusion import DiscreteDiffusion


def load_autoregressive(checkpoint_path: str, device: str):
    """Load autoregressive model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = AutoregressiveConfig(**checkpoint['model_config'])
    model = AutoregressiveCaptioner(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, model_config


def load_diffusion(checkpoint_path: str, device: str):
    """Load diffusion model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder_config = ModelConfig(**checkpoint['decoder_config'])
    model = DiffusionTransformer(decoder_config)
    model.load_state_dict(checkpoint['decoder_state_dict'])
    model.to(device)
    model.eval()

    diffusion = DiscreteDiffusion(
        vocab_size=decoder_config.vocab_size,
        mask_token_id=3,
        pad_token_id=0,
        schedule="cosine",
    )

    return model, diffusion, decoder_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive_ckpt", type=str, default="checkpoints_autoregressive_synthetic/best.pt")
    parser.add_argument("--diffusion_ckpt", type=str, default="checkpoints_caption_small/best.pt")
    parser.add_argument("--data_dir", type=str, default="data_captions_synthetic")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("Model Comparison: Autoregressive vs Diffusion")
    print("=" * 80)

    # Load config
    config_path = os.path.join(args.data_dir, "config.json")
    with open(config_path) as f:
        data_config = json.load(f)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(data_config['tokenizer'])
    print(f"Tokenizer: {len(tokenizer.get_vocab())} tokens")

    # Load autoregressive model
    print(f"\nLoading autoregressive model from: {args.autoregressive_ckpt}")
    ar_model, ar_config = load_autoregressive(args.autoregressive_ckpt, args.device)
    print(f"  Layers: {ar_config.n_layers}, d_model: {ar_config.d_model}")

    # Load diffusion model
    print(f"\nLoading diffusion model from: {args.diffusion_ckpt}")
    diff_model, diffusion, diff_config = load_diffusion(args.diffusion_ckpt, args.device)
    print(f"  Layers: {diff_config.n_layers}, d_model: {diff_config.d_model}")

    # Load validation data
    print(f"\nLoading validation data from: {args.data_dir}")
    val_features = torch.load(os.path.join(args.data_dir, "val_image_features.pt"))
    val_captions = torch.load(os.path.join(args.data_dir, "val_captions.pt"))
    print(f"Validation set: {len(val_captions)} examples")
    print("=" * 80)

    # Generate captions for first N examples
    print(f"\nComparing {args.num_examples} examples:\n")

    for i in range(min(args.num_examples, len(val_captions))):
        image_features = val_features[i:i+1].to(device)
        reference_tokens = val_captions[i].tolist()

        # Decode reference
        ref_tokens_clean = [t for t in reference_tokens if t not in [0, 1, 2]]
        reference_text = tokenizer.decode(ref_tokens_clean)

        # Generate with autoregressive model
        with torch.no_grad():
            ar_tokens = ar_model.generate(
                encoder_output=image_features,
                max_len=ar_config.max_seq_len,
                temperature=1.0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ar_tokens_clean = [t for t in ar_tokens[0].tolist() if t not in [0, 1, 2]]
        ar_text = tokenizer.decode(ar_tokens_clean)

        # Generate with diffusion model
        with torch.no_grad():
            image_mask = torch.ones(image_features.shape[:2], device=device)
            diff_tokens = diffusion.sample(
                diff_model,
                batch_size=1,
                seq_len=diff_config.max_seq_len,
                encoder_output=image_features,
                encoder_attention_mask=image_mask,
                num_steps=25,
                temperature=1.0,
                device=device,
            )
        diff_tokens_clean = [t for t in diff_tokens[0].tolist() if t not in [0, 1, 2, 3]]
        diff_text = tokenizer.decode(diff_tokens_clean)

        print(f"Example {i+1}:")
        print(f"  Reference:      {reference_text}")
        print(f"  Autoregressive: {ar_text}")
        print(f"  Diffusion:      {diff_text}")
        print()


if __name__ == "__main__":
    main()
