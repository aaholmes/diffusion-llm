#!/usr/bin/env python3
"""
Generate samples from a trained diffusion language model.

Usage:
    python generate.py                          # Default settings
    python generate.py --num_samples 10         # More samples
    python generate.py --temperature 0.7        # More focused
    python generate.py --temperature 1.2        # More creative
    python generate.py --steps 200              # Higher quality (slower)
    python generate.py --checkpoint checkpoints_long/step_10000.pt  # Different checkpoint
"""

import argparse
import torch
from tokenizers import Tokenizer

from src.core.model import DiffusionTransformer, create_model
from src.core.diffusion import DiscreteDiffusion


def generate(
    checkpoint_path: str = "checkpoints_long/final.pt",
    tokenizer_path: str = "data_full/tokenizer.json",
    num_samples: int = 5,
    seq_len: int = 128,
    steps: int = 100,
    temperature: float = 0.8,
):
    """Generate samples from trained model."""

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Create model
    model_config = checkpoint.get("model_config", "small")
    if isinstance(model_config, str):
        model = create_model(model_config, vocab_size=8192, max_seq_len=256)
    else:
        model = DiffusionTransformer(model_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Setup diffusion
    diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Generating {num_samples} samples (steps={steps}, temp={temperature})...\n")

    # Generate
    samples = diffusion.sample(
        model,
        batch_size=num_samples,
        seq_len=seq_len,
        num_steps=steps,
        temperature=temperature,
        device=device,
    )

    # Decode and print
    for i, sample in enumerate(samples):
        tokens = sample.tolist()

        # Find EOS token (id=2) and trim
        try:
            eos_idx = tokens.index(2)
            tokens = tokens[1:eos_idx]  # Skip BOS, stop at EOS
        except ValueError:
            tokens = tokens[1:]  # Skip BOS

        text = tokenizer.decode(tokens)
        print(f"{'='*60}")
        print(f"Sample {i+1}")
        print(f"{'='*60}")
        print(text)
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate samples from diffusion LM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_long/final.pt",
                        help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json",
                        help="Path to tokenizer")
    parser.add_argument("--num_samples", "-n", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of diffusion steps (more=better quality)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                        help="Sampling temperature (0.7=focused, 1.0+=creative)")

    args = parser.parse_args()

    generate(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        steps=args.steps,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
