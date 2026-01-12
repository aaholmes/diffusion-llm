#!/usr/bin/env python3
"""
Long training run configuration for TinyStories.

Designed for ~20 hour CPU run or faster on GPU.
Run with: python train_long.py

Key design decisions:
- small model (17M params): Best quality/time tradeoff for 20h CPU
- Full dataset: 2.1M stories for best generalization
- Conservative LR with long warmup: Stable training
- Regular validation: Catch overfitting early
- Weight decay: Standard regularization

Expected results (20h CPU):
- ~25,000 training steps
- ~0.4 epochs of full data
- Loss should reach ~4.5-5.0 (vs 5.7 at 500 steps)
"""

from dataclasses import dataclass


@dataclass
class LongTrainConfig:
    """Configuration for long training run."""

    # Model - small is best for 20h CPU
    model_config: str = "small"
    vocab_size: int = 8192
    max_seq_len: int = 256

    # Data - use full dataset for best generalization
    subset_size: int = None  # None = full dataset (2.1M)
    val_split: float = 0.02  # 2% = ~42k validation samples

    # Training - conservative for stability
    batch_size: int = 32
    grad_accum_steps: int = 2  # Effective batch = 64
    max_steps: int = 12500  # ~15 hours on CPU

    # Learning rate - standard transformer settings
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1  # Decay to 10% of max LR
    warmup_steps: int = 500  # 4% warmup

    # Regularization - prevent overfitting
    weight_decay: float = 0.1  # Standard for transformers
    max_grad_norm: float = 1.0
    dropout: float = 0.1  # Already in model default

    # Validation - monitor generalization
    eval_every: int = 500  # Every 500 steps
    num_eval_batches: int = 50  # Enough for stable estimate

    # Checkpointing
    save_every: int = 2500  # Save every 2500 steps (~10 checkpoints)
    keep_last_n: int = 3
    checkpoint_dir: str = "checkpoints_long"

    # Logging
    log_every: int = 25
    use_wandb: bool = False  # Set True if wandb available
    wandb_project: str = "diffusion-lm-long"

    # Early stopping - stop if val loss doesn't improve for 10 evals (5000 steps)
    early_stopping_patience: int = 10


# Hyperparameter justification:
#
# Learning Rate (3e-4):
# - Standard for AdamW with transformers
# - Higher (1e-3) can cause instability
# - Lower (1e-4) converges slower
#
# Weight Decay (0.1):
# - Standard regularization for transformers
# - Helps prevent overfitting on repeated patterns
# - Applied to all params except biases/LayerNorm
#
# Batch Size (64 effective):
# - Larger batches = more stable gradients
# - 32 fits comfortably in memory
# - 2x accumulation for effective 64
#
# Warmup (1000 steps, 4%):
# - Prevents early instability
# - Standard recommendation: 1-5% of total steps
#
# Dropout (0.1):
# - Model default, good for regularization
# - Higher (0.2+) can hurt learning
# - Lower (0.0) risks overfitting
#
# Overfitting Prevention:
# 1. Weight decay (0.1) - L2 regularization
# 2. Dropout (0.1) - prevents co-adaptation
# 3. Large dataset (2.1M) - hard to memorize
# 4. Regular validation - early warning
# 5. Cosine LR decay - smooth convergence
#
# Underfitting Prevention:
# 1. Sufficient model size (17M params)
# 2. Long enough training (25k steps)
# 3. Proper LR (not too low)
# 4. Adequate warmup (stable start)


def print_config():
    """Print configuration summary."""
    config = LongTrainConfig()

    print("=" * 60)
    print("Long Training Configuration")
    print("=" * 60)

    sections = {
        "Model": ["model_config", "vocab_size", "max_seq_len"],
        "Data": ["subset_size", "val_split"],
        "Training": ["batch_size", "grad_accum_steps", "max_steps"],
        "Learning Rate": ["learning_rate", "min_lr_ratio", "warmup_steps"],
        "Regularization": ["weight_decay", "max_grad_norm", "dropout"],
        "Validation": ["eval_every", "num_eval_batches"],
        "Checkpointing": ["save_every", "keep_last_n", "checkpoint_dir"],
    }

    for section, keys in sections.items():
        print(f"\n{section}:")
        for key in keys:
            value = getattr(config, key)
            if value is None:
                value = "Full dataset (~2.1M)"
            print(f"  {key}: {value}")

    # Compute derived values
    effective_batch = config.batch_size * config.grad_accum_steps
    total_samples = config.max_steps * effective_batch

    print(f"\nDerived:")
    print(f"  effective_batch_size: {effective_batch}")
    print(f"  total_samples_seen: {total_samples:,}")
    print(f"  estimated_time_cpu: ~20 hours")


if __name__ == "__main__":
    print_config()
