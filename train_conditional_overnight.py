#!/usr/bin/env python3
"""
Overnight CPU training run for conditional diffusion model.

Designed for ~8 hours of CPU training to get preliminary results
before GPU arrives.

Usage:
    python train_conditional_overnight.py

    # Or with nohup for true overnight run:
    nohup python train_conditional_overnight.py > overnight.log 2>&1 &
"""

import subprocess
import sys


def main():
    # Calculate training parameters for ~8 hour run
    # From proof of concept: ~1 step/second on CPU
    # 8 hours = 8 * 60 * 60 = 28,800 seconds
    # Target: ~25,000 steps (with some margin)

    cmd = [
        sys.executable, "train_conditional.py",
        "--denoiser", "checkpoints_long/final.pt",
        "--data_dir", "data_conditional",
        "--max_steps", "25000",
        "--batch_size", "16",          # Smaller batch for CPU memory
        "--learning_rate", "1e-4",
        "--warmup_steps", "1000",
        "--eval_every", "500",          # Eval every ~8 minutes
        "--save_every", "2500",         # Save every ~40 minutes
        "--log_every", "50",            # Log every ~50 seconds
        "--checkpoint_dir", "checkpoints_conditional_overnight",
        "--no_amp",                     # Disable AMP on CPU
    ]

    print("=" * 60)
    print("Overnight Conditional Training Run")
    print("=" * 60)
    print(f"Target steps: 25,000")
    print(f"Estimated time: ~7-8 hours")
    print(f"Batch size: 16")
    print(f"Learning rate: 1e-4")
    print(f"Checkpoints saved to: checkpoints_conditional_overnight/")
    print("=" * 60)
    print()
    print("Command:", " ".join(cmd))
    print()
    print("Starting training...")
    print("=" * 60)

    # Run training
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
