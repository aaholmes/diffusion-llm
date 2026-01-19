#!/usr/bin/env python3
"""
Long training run script for TinyStories.

This script handles the full pipeline:
1. Data preparation (full dataset)
2. Training with monitoring
3. Sample generation for quality check

Usage:
    python train_long.py                    # Full 20-hour run
    python train_long.py --max_steps 1000   # Quick test
    python train_long.py --resume           # Resume from checkpoint

Expected runtime: ~15 hours on CPU (12.5k steps), ~30 min on GPU
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Optimize CPU threading - 24 threads is optimal for this workload
# (32 causes contention, 16 leaves performance on the table)
torch.set_num_threads(24)

from src.data.data_prep import DataConfig, main as prepare_data, load_stories, train_tokenizer, tokenize_dataset
from src.training.train import TrainConfig, Trainer
from src.training.train_config_long import LongTrainConfig


def prepare_full_dataset(config: LongTrainConfig, force: bool = False):
    """Prepare full TinyStories dataset."""
    data_dir = "data_full"
    train_path = f"{data_dir}/train_tokens.pt"
    val_path = f"{data_dir}/val_tokens.pt"
    tokenizer_path = f"{data_dir}/tokenizer.json"

    # Check if already prepared
    if not force and all(os.path.exists(p) for p in [train_path, val_path, tokenizer_path]):
        print("Full dataset already prepared, skipping...")
        return train_path, val_path, tokenizer_path

    print("=" * 60)
    print("Preparing Full TinyStories Dataset")
    print("=" * 60)
    print("This may take 30-60 minutes for 2.1M stories...")
    print()

    data_config = DataConfig(
        subset_size=config.subset_size,  # None = full
        val_split=config.val_split,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        train_data_path=train_path,
        val_data_path=val_path,
        config_path=f"{data_dir}/config.json",
    )

    prepare_data(data_config)

    return train_path, val_path, tokenizer_path


def create_train_config(
    long_config: LongTrainConfig,
    train_path: str,
    val_path: str,
    max_steps: int = None,
    resume_from: str = None,
) -> TrainConfig:
    """Create TrainConfig from LongTrainConfig."""
    return TrainConfig(
        # Model
        model_config=long_config.model_config,
        vocab_size=long_config.vocab_size,
        max_seq_len=long_config.max_seq_len,
        # Data
        train_data_path=train_path,
        val_data_path=val_path,
        # Training
        batch_size=long_config.batch_size,
        grad_accum_steps=long_config.grad_accum_steps,
        max_steps=max_steps or long_config.max_steps,
        learning_rate=long_config.learning_rate,
        min_lr_ratio=long_config.min_lr_ratio,
        warmup_steps=long_config.warmup_steps,
        weight_decay=long_config.weight_decay,
        max_grad_norm=long_config.max_grad_norm,
        # Validation
        eval_every=long_config.eval_every,
        num_eval_batches=long_config.num_eval_batches,
        # Checkpointing
        save_every=long_config.save_every,
        keep_last_n=long_config.keep_last_n,
        checkpoint_dir=long_config.checkpoint_dir,
        # Logging
        log_every=long_config.log_every,
        use_wandb=long_config.use_wandb,
        wandb_project=long_config.wandb_project,
        # Resume
        resume_from=resume_from,
        # Early stopping
        early_stopping_patience=long_config.early_stopping_patience,
    )


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Look for step_*.pt files
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        # Try final.pt
        final = checkpoint_dir / "final.pt"
        if final.exists():
            return str(final)
        return None

    # Sort by step number
    def get_step(p):
        try:
            return int(p.stem.split("_")[1])
        except:
            return 0

    latest = max(checkpoints, key=get_step)
    return str(latest)


def generate_samples(checkpoint_path: str, tokenizer_path: str, num_samples: int = 3):
    """Generate samples from trained model."""
    from tokenizers import Tokenizer
    from src.core.model import create_model
    from src.core.diffusion import DiscreteDiffusion

    print("\n" + "=" * 60)
    print("Generating Samples")
    print("=" * 60)

    # Load
    tokenizer = Tokenizer.from_file(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model_config = checkpoint.get("model_config", "small")
    # Handle both string config names and ModelConfig objects
    if isinstance(model_config, str):
        model = create_model(model_config, vocab_size=8192, max_seq_len=256)
    else:
        # It's already a ModelConfig object, use it directly
        from src.core.model import DiffusionTransformer
        model = DiffusionTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

    # Generate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    samples = diffusion.sample(
        model,
        batch_size=num_samples,
        seq_len=128,
        num_steps=100,
        temperature=0.8,
        device=device,
    )

    print()
    for i, sample in enumerate(samples):
        tokens = sample.tolist()
        # Find EOS
        try:
            eos_idx = tokens.index(2)
            tokens = tokens[1:eos_idx]  # Skip BOS, stop at EOS
        except ValueError:
            tokens = tokens[1:]  # Skip BOS

        text = tokenizer.decode(tokens)
        print(f"Sample {i+1}:")
        print(f"  {text[:200]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Long training run for TinyStories")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps (default: 25000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate samples from checkpoint")
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test with 100 steps")
    args = parser.parse_args()

    # Load config
    config = LongTrainConfig()

    # Quick test mode
    if args.quick_test:
        args.max_steps = 100
        config.eval_every = 50
        config.save_every = 50
        config.subset_size = 1000  # Use tiny subset for quick test

    # Paths
    data_dir = "data_full"
    train_path = f"{data_dir}/train_tokens.pt"
    val_path = f"{data_dir}/val_tokens.pt"
    tokenizer_path = f"{data_dir}/tokenizer.json"

    # Generate only mode
    if args.generate_only:
        checkpoint = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint:
            generate_samples(checkpoint, tokenizer_path)
        else:
            print("No checkpoint found!")
        return

    # Prepare data
    if not args.skip_data_prep:
        train_path, val_path, tokenizer_path = prepare_full_dataset(config)

    # Find resume checkpoint
    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(config.checkpoint_dir)
        if resume_from:
            print(f"Resuming from: {resume_from}")
        else:
            print("No checkpoint found, starting fresh")

    # Create training config
    train_config = create_train_config(
        config,
        train_path,
        val_path,
        max_steps=args.max_steps,
        resume_from=resume_from,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model: {train_config.model_config}")
    print(f"Max steps: {train_config.max_steps:,}")
    print(f"Effective batch size: {train_config.batch_size * train_config.grad_accum_steps}")
    print(f"Learning rate: {train_config.learning_rate}")
    print(f"Weight decay: {train_config.weight_decay}")
    print(f"Warmup steps: {train_config.warmup_steps}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Estimate time
    if not torch.cuda.is_available():
        # CPU estimate based on small model: ~4.3s/step (full training loop)
        est_hours = (train_config.max_steps * 4.3) / 3600
        print(f"Estimated time (CPU): ~{est_hours:.1f} hours")
    print()

    # Train
    start_time = time.time()
    trainer = Trainer(train_config)
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTotal training time: {elapsed/3600:.2f} hours")

    # Generate samples
    checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    if checkpoint:
        generate_samples(checkpoint, tokenizer_path)


if __name__ == "__main__":
    main()
