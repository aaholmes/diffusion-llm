#!/usr/bin/env python3
"""
Train autoregressive image captioning model (baseline).

Usage:
    python train_autoregressive.py --data_dir data_captions_synthetic
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.model_autoregressive import AutoregressiveCaptioner, AutoregressiveConfig


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data_captions_synthetic"
    max_caption_len: int = 128

    # Model
    d_model: int = 768
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 50
    checkpoint_dir: str = "checkpoints_autoregressive"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True


class CaptionDataset(Dataset):
    """Dataset of (image_features, caption_tokens)."""

    def __init__(self, image_features: torch.Tensor, caption_tokens: torch.Tensor):
        self.image_features = image_features
        self.caption_tokens = caption_tokens

    def __len__(self):
        return len(self.caption_tokens)

    def __getitem__(self, idx):
        return {
            'image_features': self.image_features[idx],
            'caption_tokens': self.caption_tokens[idx],
        }


class AutoregressiveTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_model(self):
        """Initialize model."""
        print("=" * 60)
        print("Setting up autoregressive captioning model")
        print("=" * 60)

        # Load data config
        config_path = os.path.join(self.config.data_dir, "config.json")
        with open(config_path) as f:
            data_config = json.load(f)

        feature_dim = data_config['feature_dim']
        vocab_size = data_config['vocab_size']

        print(f"\nImage features: d_model={feature_dim}")

        # Create model config
        model_config = AutoregressiveConfig(
            d_model=feature_dim,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            vocab_size=vocab_size,
            max_seq_len=self.config.max_caption_len,
            dropout=self.config.dropout,
            pad_token_id=0,
        )

        self.model = AutoregressiveCaptioner(model_config)
        self.model.to(self.device)

        print(f"\nModel: {model_config.n_layers} layers, d_model={model_config.d_model}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)

        self.model_config = model_config

    def _setup_data(self):
        """Load data."""
        print("\nLoading data...")

        train_features = torch.load(os.path.join(self.config.data_dir, "train_image_features.pt"))
        train_captions = torch.load(os.path.join(self.config.data_dir, "train_captions.pt"))

        val_features = torch.load(os.path.join(self.config.data_dir, "val_image_features.pt"))
        val_captions = torch.load(os.path.join(self.config.data_dir, "val_captions.pt"))

        print(f"  Train: {len(train_captions):,} examples")
        print(f"  Val: {len(val_captions):,} examples")

        train_dataset = CaptionDataset(train_features, train_captions)
        val_dataset = CaptionDataset(val_features, val_captions)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp)

    def get_lr(self, step: int) -> float:
        """Learning rate schedule with warmup and cosine decay."""
        max_lr = self.config.learning_rate
        min_lr = max_lr * 0.1

        # Warmup
        if step < self.config.warmup_steps:
            return max_lr * step / self.config.warmup_steps

        # Cosine decay
        progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine_decay

    def train_step(self, batch):
        """Single training step."""
        self.model.train()

        image_features = batch['image_features'].to(self.device)
        captions = batch['caption_tokens'].to(self.device)

        # Create attention masks
        image_mask = torch.ones(image_features.shape[:2], device=self.device)

        # Get input (all but last token) and target (all but first token)
        input_tokens = captions[:, :-1]
        target_tokens = captions[:, 1:]

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            logits = self.model(input_tokens, image_features, image_mask)

            # Compute loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, self.model_config.vocab_size),
                target_tokens.reshape(-1),
                ignore_index=0,  # pad_token_id
                reduction='mean'
            )

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Calculate accuracy (on non-padding tokens)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = (target_tokens != 0)
            correct = (preds == target_tokens) & mask
            accuracy = correct.sum().float() / mask.sum().float()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
        }

    @torch.no_grad()
    def validate(self):
        """Validation loop."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in self.val_loader:
            image_features = batch['image_features'].to(self.device)
            captions = batch['caption_tokens'].to(self.device)

            image_mask = torch.ones(image_features.shape[:2], device=self.device)

            input_tokens = captions[:, :-1]
            target_tokens = captions[:, 1:]

            logits = self.model(input_tokens, image_features, image_mask)

            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.model_config.vocab_size),
                target_tokens.reshape(-1),
                ignore_index=0,
                reduction='sum'
            )

            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = (target_tokens != 0)
            correct = ((preds == target_tokens) & mask).sum()

            total_loss += loss.item()
            total_correct += correct.item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / total_tokens
        avg_acc = total_correct / total_tokens

        return avg_loss, avg_acc

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'global_step': step,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config.__dict__,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_config': {
                'data_dir': self.config.data_dir,
                'max_caption_len': self.config.max_caption_len,
                'learning_rate': self.config.learning_rate,
            }
        }

        path = os.path.join(self.config.checkpoint_dir, f"step_{step}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Autoregressive Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print("=" * 60)

        train_iter = iter(self.train_loader)
        pbar = tqdm(range(self.config.max_steps), desc="Training")

        for step in pbar:
            self.global_step = step

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Update learning rate
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Training step
            metrics = self.train_step(batch)

            # Log
            if step % self.config.log_every == 0:
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    acc=f"{metrics['accuracy']:.3f}",
                    lr=f"{lr:.2e}"
                )
                print(f"Step {step}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.3f}, lr={lr:.2e}")

            # Validation
            if (step + 1) % self.config.eval_every == 0:
                val_loss, val_acc = self.validate()
                print(f"\n[Val] loss={val_loss:.4f}, acc={val_acc:.3f}")

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print("  New best model!")

                # Save checkpoint
                if (step + 1) % self.config.save_every == 0:
                    self.save_checkpoint(step + 1, is_best)

        # Final save
        self.save_checkpoint(self.config.max_steps, is_best=False)

        # Save final model
        final_path = os.path.join(self.config.checkpoint_dir, "final.pt")
        checkpoint = {
            'global_step': self.config.max_steps,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config.__dict__,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, final_path)

        print("\n" + "=" * 60)
        print("Autoregressive Training Complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=str, default="data_captions_synthetic")

    # Model
    parser.add_argument("--n_layers", type=int, default=6)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_autoregressive")
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
    )

    trainer = AutoregressiveTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
