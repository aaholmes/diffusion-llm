#!/usr/bin/env python3
"""
Training script for diffusion language model.

Implements a complete training pipeline with:
- Mixed precision training (FP16)
- Learning rate scheduling (warmup + cosine decay)
- Gradient accumulation
- Checkpointing
- Wandb logging (optional)
- Validation loop

Usage:
    python train.py                           # Train with defaults
    python train.py --model_config small      # Train small model
    python train.py --max_steps 10000         # Custom training length
    python train.py --no_wandb                # Disable wandb logging
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.core.model import create_model, DiffusionTransformer, ModelConfig
from src.core.diffusion import DiscreteDiffusion


@dataclass
class TrainConfig:
    """Configuration for training."""
    # Model
    model_config: str = "small"
    vocab_size: int = 8192
    max_seq_len: int = 256

    # Data
    train_data_path: str = "data/train_tokens.pt"
    val_data_path: str = "data/val_tokens.pt"
    tokenizer_path: str = "data/tokenizer.json"

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 50000

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # LR Schedule
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of max

    # Validation & Checkpointing
    eval_every: int = 500
    save_every: int = 5000
    num_eval_batches: int = 50
    checkpoint_dir: str = "checkpoints"
    keep_last_n: int = 3

    # Logging
    log_every: int = 10
    use_wandb: bool = True
    wandb_project: str = "diffusion-lm"
    wandb_run_name: Optional[str] = None

    # Hardware
    device: str = "cuda"
    use_amp: bool = True  # Automatic mixed precision
    num_workers: int = 4

    # Diffusion
    mask_token_id: int = 3
    pad_token_id: int = 0

    # Resume
    resume_from: Optional[str] = None

    # Early stopping (patience=0 disables)
    early_stopping_patience: int = 0

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


class Trainer:
    """
    Trainer for diffusion language model.

    Handles the complete training loop including:
    - Data loading
    - Optimization with mixed precision
    - LR scheduling
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Setup
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_logging()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.evals_without_improvement = 0

        # Resume if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _setup_model(self):
        """Initialize model and diffusion process."""
        print(f"Creating {self.config.model_config} model...")

        self.model = create_model(
            self.config.model_config,
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            pad_token_id=self.config.pad_token_id,
        )
        self.model.to(self.device)

        # Compile model for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.diffusion = DiscreteDiffusion(
            vocab_size=self.config.vocab_size,
            mask_token_id=self.config.mask_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def _setup_data(self):
        """Load and prepare data."""
        print("Loading training data...")
        train_tokens = torch.load(self.config.train_data_path, weights_only=True)
        self.train_dataset = TensorDataset(train_tokens)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"  Train: {len(self.train_dataset):,} sequences")

        print("Loading validation data...")
        val_tokens = torch.load(self.config.val_data_path, weights_only=True)
        self.val_dataset = TensorDataset(val_tokens)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        print(f"  Val: {len(self.val_dataset):,} sequences")

    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            eps=self.config.eps,
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp)

    def _setup_logging(self):
        """Initialize logging (wandb or console)."""
        self.use_wandb = self.config.use_wandb

        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=asdict(self.config),
                )
                print(f"Wandb initialized: {wandb.run.url}")
            except ImportError:
                print("Wandb not installed, disabling...")
                self.use_wandb = False
            except Exception as e:
                print(f"Wandb init failed: {e}, disabling...")
                self.use_wandb = False

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Setup CSV logging for metrics
        self.metrics_path = os.path.join(self.config.checkpoint_dir, "metrics.csv")
        self._last_train_metrics = {}  # Will be populated during training
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write("step,train_loss,train_acc,val_loss,val_acc,lr,grad_norm\n")

    def get_lr(self, step: int) -> float:
        """
        Compute learning rate with linear warmup and cosine decay.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        max_lr = self.config.learning_rate
        min_lr = max_lr * self.config.min_lr_ratio
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        if step < warmup_steps:
            # Linear warmup
            return max_lr * step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (max_lr - min_lr) * cosine_decay

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Token tensor [batch_size, seq_len]

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            loss, metrics = self.diffusion.training_losses(self.model, batch)
            loss = loss / self.config.grad_accum_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        return metrics

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return grad_norm.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run validation loop.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch in self.val_loader:
            if num_batches >= self.config.num_eval_batches:
                break

            tokens = batch[0].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                loss, metrics = self.diffusion.training_losses(self.model, tokens)

            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_accuracy / num_batches,
        }

    def save_checkpoint(self, name: str = None, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name (default: step number)
            is_best: Whether this is the best model so far
        """
        if name is None:
            name = f"step_{self.global_step}"

        # Get model state (handle compiled models)
        if hasattr(self.model, '_orig_mod'):
            model_state = self.model._orig_mod.state_dict()
            model_config = self.model._orig_mod.config
        else:
            model_state = self.model.state_dict()
            model_config = self.model.config

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "model_config": model_config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_config": asdict(self.config),
            "best_val_loss": self.best_val_loss,
        }

        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        # Keep last N + best
        while len(checkpoints) > self.config.keep_last_n:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model state
        if hasattr(self.model, '_orig_mod'):
            self.model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scaler state
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore training state
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        print(f"Resumed from step {self.global_step}")

    def train(self):
        """Run the main training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Model: {self.config.model_config}")
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

        # Training state
        self.optimizer.zero_grad()
        accum_metrics = {"loss": 0, "accuracy": 0, "mask_rate": 0}
        accum_count = 0
        tokens_processed = 0
        start_time = time.time()

        # Progress bar
        pbar = tqdm(
            total=self.config.max_steps,
            initial=self.global_step,
            desc="Training",
            unit="step"
        )

        # Create infinite data iterator
        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            # Get batch (reset iterator if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            tokens = batch[0]
            tokens_processed += tokens.numel()

            # Update learning rate
            lr = self.get_lr(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Training step
            metrics = self.train_step(tokens)

            # Accumulate metrics
            for key in accum_metrics:
                accum_metrics[key] += metrics.get(key, 0)
            accum_count += 1

            # Optimizer step (with gradient accumulation)
            if accum_count >= self.config.grad_accum_steps:
                grad_norm = self.optimizer_step()
                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.config.log_every == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed

                    log_metrics = {
                        "train/loss": accum_metrics["loss"] / accum_count,
                        "train/accuracy": accum_metrics["accuracy"] / accum_count,
                        "train/mask_rate": accum_metrics["mask_rate"] / accum_count,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": self.global_step,
                    }

                    # Update progress bar
                    pbar.set_postfix(
                        loss=f"{log_metrics['train/loss']:.4f}",
                        acc=f"{log_metrics['train/accuracy']:.3f}",
                        lr=f"{lr:.2e}"
                    )

                    # Print metrics (useful for log files)
                    print(f"Step {self.global_step:>6}: loss={log_metrics['train/loss']:.4f}, "
                          f"acc={log_metrics['train/accuracy']:.3f}, lr={lr:.2e}, "
                          f"grad_norm={grad_norm:.2f}, tok/s={tokens_per_sec:.0f}")

                    # Log to CSV (val columns filled in later if this is a val step)
                    self._last_train_metrics = {
                        "step": self.global_step,
                        "train_loss": log_metrics['train/loss'],
                        "train_acc": log_metrics['train/accuracy'],
                        "lr": lr,
                        "grad_norm": grad_norm,
                    }

                    # Log to wandb
                    if self.use_wandb:
                        import wandb
                        wandb.log(log_metrics, step=self.global_step)

                    # Write to CSV if not a validation step (val steps write below)
                    if self.global_step % self.config.eval_every != 0:
                        with open(self.metrics_path, "a") as f:
                            m = self._last_train_metrics
                            f.write(f"{m['step']},{m['train_loss']:.6f},{m['train_acc']:.6f},,,"
                                    f"{m['lr']:.2e},{m['grad_norm']:.4f}\n")

                # Reset accumulation
                accum_metrics = {"loss": 0, "accuracy": 0, "mask_rate": 0}
                accum_count = 0

                # Validation
                if self.global_step % self.config.eval_every == 0:
                    val_metrics = self.evaluate()

                    print(f"\nStep {self.global_step}: "
                          f"val_loss={val_metrics['val_loss']:.4f}, "
                          f"val_acc={val_metrics['val_accuracy']:.3f}")

                    # Check for best model
                    is_best = val_metrics["val_loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.evals_without_improvement = 0
                        print(f"  New best model!")
                    else:
                        self.evals_without_improvement += 1
                        if self.config.early_stopping_patience > 0:
                            print(f"  No improvement ({self.evals_without_improvement}/{self.config.early_stopping_patience})")

                    # Early stopping check
                    if (self.config.early_stopping_patience > 0 and
                        self.evals_without_improvement >= self.config.early_stopping_patience):
                        print(f"\nEarly stopping: no improvement for {self.config.early_stopping_patience} evaluations")
                        break

                    # Log to wandb
                    if self.use_wandb:
                        import wandb
                        wandb.log(val_metrics, step=self.global_step)

                    # Write to CSV with validation metrics
                    with open(self.metrics_path, "a") as f:
                        m = self._last_train_metrics
                        f.write(f"{m['step']},{m['train_loss']:.6f},{m['train_acc']:.6f},"
                                f"{val_metrics['val_loss']:.6f},{val_metrics['val_accuracy']:.6f},"
                                f"{m['lr']:.2e},{m['grad_norm']:.4f}\n")

                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

        pbar.close()

        # Final checkpoint
        self.save_checkpoint("final")
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        if self.use_wandb:
            import wandb
            wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train diffusion language model")

    # Model
    parser.add_argument("--model_config", type=str, default="small",
                       choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--max_seq_len", type=int, default=256)

    # Data
    parser.add_argument("--train_data_path", type=str, default="data/train_tokens.pt")
    parser.add_argument("--val_data_path", type=str, default="data/val_tokens.pt")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # Validation & Checkpointing
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Logging
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="diffusion-lm")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    # Create config
    config = TrainConfig(
        model_config=args.model_config,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume_from=args.resume_from,
    )

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
