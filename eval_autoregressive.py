#!/usr/bin/env python3
"""
Evaluate autoregressive captioning model.
"""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from model_autoregressive import AutoregressiveCaptioner, AutoregressiveConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model from config
    model_config = AutoregressiveConfig(**checkpoint['model_config'])
    model = AutoregressiveCaptioner(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_autoregressive/best.pt")
    parser.add_argument("--data_dir", type=str, default="data_captions_synthetic")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Autoregressive Captioning Evaluation")
    print("=" * 60)

    # Load config to get tokenizer path
    config_path = os.path.join(args.data_dir, "config.json")
    with open(config_path) as f:
        data_config = json.load(f)

    # Load tokenizer
    tokenizer_path = data_config['tokenizer']
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded tokenizer: {len(tokenizer.get_vocab())} tokens")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_config = load_model(args.checkpoint, args.device)
    print(f"Model: {model_config.n_layers} layers, vocab={model_config.vocab_size}")

    # Load validation data
    print(f"\nLoading validation data from: {args.data_dir}")
    val_features = torch.load(os.path.join(args.data_dir, "val_image_features.pt"))
    val_captions = torch.load(os.path.join(args.data_dir, "val_captions.pt"))

    print(f"Validation set: {len(val_captions)} examples")
    print("=" * 60)

    # Generate captions for first N examples
    print(f"\nGenerating {args.num_examples} examples:\n")

    for i in range(min(args.num_examples, len(val_captions))):
        image_features = val_features[i:i+1].to(device)
        reference_tokens = val_captions[i].tolist()

        # Decode reference
        # Remove BOS, EOS, and padding
        ref_tokens_clean = [t for t in reference_tokens if t not in [0, 1, 2]]
        reference_text = tokenizer.decode(ref_tokens_clean)

        # Generate caption
        with torch.no_grad():
            generated_tokens = model.generate(
                encoder_output=image_features,
                max_len=model_config.max_seq_len,
                temperature=args.temperature,
                bos_token_id=1,
                eos_token_id=2,
            )

        # Decode generated
        gen_tokens = generated_tokens[0].tolist()
        gen_tokens_clean = [t for t in gen_tokens if t not in [0, 1, 2]]
        generated_text = tokenizer.decode(gen_tokens_clean)

        print(f"Example {i+1}:")
        print(f"  Reference: {reference_text}")
        print(f"  Generated: {generated_text}")
        print()


if __name__ == "__main__":
    main()
