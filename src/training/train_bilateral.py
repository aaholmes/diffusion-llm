#!/usr/bin/env python3
"""
Training script for Sparse Distribution Diffusion on TinyStories.

Text-only unconditional generation using the v2 architecture with:
- Intra-position attention (k×k per position with prob bias)
- Inter-position attention (L×L pooled)
- Learned readout for k→1 collapse

Usage:
    python train_bilateral.py --max_steps 20000
    python train_bilateral.py --model_version v1 --max_steps 20000  # Compare with v1
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.core.sparse_diffusion import SparseDiffusion, SDDConfig, SparseState
from src.core.sparse_model import (
    SparseDenoiser, SparseModelConfig, create_sparse_model,
    BilateralSparseDenoiser, create_bilateral_sparse_model,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_dir: str = "data_full"
    checkpoint_dir: str = "checkpoints_bilateral"

    # Model
    model_version: str = "v2"  # "v1" (SparseDenoiser) or "v2" (BilateralSparseDenoiser)
    embed_dim: int = 64
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 256

    # Curriculum for k
    k_schedule: tuple = (1, 2, 4, 8)  # k values to use
    k_warmup_steps: int = 2500  # Steps before increasing k

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 20000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Logging
    log_every: int = 50
    val_every: int = 500
    save_every: int = 2000
    generate_every: int = 1000  # Generate samples

    # Mixed precision
    use_amp: bool = True


class TextDataset(Dataset):
    """Dataset for tokenized text sequences."""

    def __init__(self, tokens_path: str, max_samples: Optional[int] = None):
        self.tokens = torch.load(tokens_path)
        if max_samples is not None:
            self.tokens = self.tokens[:max_samples]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


class BilateralTrainer:
    """Trainer for Sparse Distribution Diffusion on text."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data config
        data_config_path = os.path.join(config.data_dir, "config.json")
        with open(data_config_path) as f:
            self.data_config = json.load(f)

        # Get vocab size from data config
        vocab_size = self.data_config["vocab_size"]

        # Create model based on version
        # Use max k from schedule for model creation to support k-curriculum
        max_k = max(config.k_schedule)
        if config.model_version == "v2":
            self.model, self.sdd_config = create_bilateral_sparse_model(
                vocab_size=vocab_size,
                embed_dim=config.embed_dim,
                k=max_k,  # Use max k to support curriculum
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                encoder_dim=None,  # No encoder for text-only
                max_seq_len=config.max_seq_len,
            )
            # Set initial k for curriculum
            self.model.config.k = config.k_schedule[0]
            self.sdd_config.k = config.k_schedule[0]
            print(f"Created BilateralSparseDenoiser (v2)")
        else:
            self.model, self.sdd_config = create_sparse_model(
                vocab_size=vocab_size,
                embed_dim=config.embed_dim,
                k=config.k_schedule[0],
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                encoder_dim=None,  # No encoder for text-only
                max_seq_len=config.max_seq_len,
            )
            print(f"Created SparseDenoiser (v1)")

        self.model.to(self.device)

        # Create diffusion process
        self.diffusion = SparseDiffusion(
            self.sdd_config,
            self.model.get_embedding_table(),
        )

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if config.use_amp else None

        # Load data
        self.train_loader = self._create_dataloader("train")
        self.val_loader = self._create_dataloader("val")

        # Load tokenizer for generation
        from tokenizers import Tokenizer
        tokenizer_path = os.path.join(config.data_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # State
        self.global_step = 0
        self.current_k = config.k_schedule[0]
        self.best_val_loss = float("inf")

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create dataloader for a split."""
        tokens_path = os.path.join(self.config.data_dir, f"{split}_tokens.pt")

        # Limit validation set for speed
        max_samples = None if split == "train" else 5000

        dataset = TextDataset(tokens_path, max_samples=max_samples)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )

    def get_current_k(self) -> int:
        """Get current k value based on curriculum."""
        k_schedule = self.config.k_schedule
        warmup = self.config.k_warmup_steps

        k_idx = min(self.global_step // warmup, len(k_schedule) - 1)
        return k_schedule[k_idx]

    def update_k_if_needed(self):
        """Update k value if needed."""
        new_k = self.get_current_k()
        if new_k != self.current_k:
            print(f"\n[Curriculum] Increasing k: {self.current_k} -> {new_k}")
            self.current_k = new_k
            self.model.config.k = new_k
            self.sdd_config.k = new_k

    def get_lr(self) -> float:
        """Get learning rate with warmup and cosine decay."""
        step = self.global_step
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        max_lr = self.config.learning_rate
        min_lr = max_lr * 0.1

        if step < warmup:
            return max_lr * step / warmup

        progress = (step - warmup) / (max_steps - warmup)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return min_lr + (max_lr - min_lr) * cosine_decay

    def train_step(self, tokens: torch.Tensor) -> dict:
        """Single training step."""
        self.model.train()

        tokens = tokens.to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=self.config.use_amp):
            loss, metrics = self.diffusion.training_step(
                self.model,
                tokens,
                encoder_output=None,  # No conditioning
                encoder_mask=None,
            )

        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for tokens in self.val_loader:
            tokens = tokens.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                loss, metrics = self.diffusion.training_step(
                    self.model,
                    tokens,
                    encoder_output=None,
                    encoder_mask=None,
                )

            batch_size = tokens.shape[0]
            total_loss += loss.item() * batch_size

            # Count correct predictions
            mask = tokens != 0  # non-padding
            total_tokens += mask.sum().item()
            total_correct += metrics["accuracy"] * mask.sum().item()
            num_batches += 1

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0

        return {"val_loss": avg_loss, "val_acc": avg_acc}

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 4,
        seq_len: int = 64,
        temperature: float = 0.8,
    ) -> list:
        """Generate sample texts.

        Args:
            num_samples: Number of samples to generate
            seq_len: Sequence length
            temperature: Sampling temperature (lower = more confident)
        """
        self.model.eval()

        tokens, _ = self.diffusion.sample(
            self.model,
            batch_size=num_samples,
            seq_len=seq_len,
            num_steps=25,
            temperature=temperature,
            encoder_output=None,
            encoder_mask=None,
            device=self.device,
        )

        # Decode tokens
        texts = []
        for i in range(num_samples):
            token_list = tokens[i].cpu().tolist()
            # Filter special tokens
            filtered = [t for t in token_list if t not in [0, 1, 2, 3, 4, 5]]
            text = self.tokenizer.decode(filtered)
            texts.append(text)

        return texts

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": asdict(self.model.config),
            "sdd_config": asdict(self.sdd_config),
            "train_config": asdict(self.config),
            "current_k": self.current_k,
            "best_val_loss": self.best_val_loss,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save step checkpoint
        path = os.path.join(
            self.config.checkpoint_dir, f"step_{self.global_step}.pt"
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print(f"Training SDD {self.config.model_version.upper()} on TinyStories")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"k schedule: {self.config.k_schedule}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print("=" * 60)

        train_iter = iter(self.train_loader)
        pbar = tqdm(range(self.config.max_steps), desc="Training")

        for step in pbar:
            self.global_step = step

            # Update learning rate
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Check curriculum
            self.update_k_if_needed()

            # Get batch
            try:
                tokens = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                tokens = next(train_iter)

            # Training step
            metrics = self.train_step(tokens)

            # Update progress bar
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                acc=f"{metrics['accuracy']:.3f}",
                k=self.current_k,
                lr=f"{lr:.2e}",
            )

            # Logging
            if (step + 1) % self.config.log_every == 0:
                log_str = (
                    f"\nStep {step+1}: loss={metrics['loss']:.4f}, "
                    f"acc={metrics['accuracy']:.3f}, "
                    f"ignore_acc={metrics.get('ignore_acc', 0):.3f}, "
                    f"k={self.current_k}"
                )
                print(log_str)

            # Generate samples
            if (step + 1) % self.config.generate_every == 0:
                print("\n--- Generated Samples ---")
                samples = self.generate_samples(num_samples=2, seq_len=64)
                for i, text in enumerate(samples):
                    print(f"[{i+1}] {text[:200]}...")
                print("-------------------------\n")

            # Validation
            if (step + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                print(
                    f"\n[Val] loss={val_metrics['val_loss']:.4f}, "
                    f"acc={val_metrics['val_acc']:.3f}"
                )

                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    print("  New best model!")
                    self.save_checkpoint(is_best=True)

            # Save checkpoint
            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()

        # Final generation
        print("\n" + "=" * 60)
        print("Final Generated Samples")
        print("=" * 60)
        samples = self.generate_samples(num_samples=4, seq_len=128)
        for i, text in enumerate(samples):
            print(f"\n[{i+1}] {text}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Distribution Diffusion")
    parser.add_argument("--data_dir", type=str, default="data_full")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_bilateral")
    parser.add_argument("--model_version", type=str, default="v2",
                        choices=["v1", "v2"], help="v1=SparseDenoiser, v2=BilateralSparseDenoiser")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--k_schedule", type=str, default="1,2,4,8",
                        help="Comma-separated k values for curriculum")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--generate_every", type=int, default=1000)

    args = parser.parse_args()

    # Parse k schedule
    k_schedule = tuple(int(k) for k in args.k_schedule.split(","))

    config = TrainConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_version=args.model_version,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        k_schedule=k_schedule,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        generate_every=args.generate_every,
    )

    trainer = BilateralTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
