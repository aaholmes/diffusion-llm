#!/usr/bin/env python3
"""
Train image captioning model using discrete diffusion.

Architecture:
- Frozen CLIP vision features (pre-extracted)
- Diffusion decoder with cross-attention
- Training only the decoder (CLIP features are frozen)

Usage:
    python train_captioning.py --data_dir data_captions_synthetic --checkpoint_dir checkpoints_caption
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import DiffusionTransformer, ModelConfig, MODEL_CONFIGS
from diffusion import DiscreteDiffusion


@dataclass
class CaptionTrainConfig:
    # Data
    data_dir: str = "data_captions_synthetic"
    max_caption_len: int = 128

    # Model - decoder only (encoder features pre-extracted)
    decoder_config: str = "small"
    decoder_n_layers: int = 6

    # Training
    batch_size: int = 64
    max_steps: int = 5000
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    eval_every: int = 200
    save_every: int = 1000
    log_every: int = 50
    checkpoint_dir: str = "checkpoints_caption"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True


class CaptionDataset(Dataset):
    """Dataset of (image_features, caption_tokens)."""

    def __init__(self, image_features: torch.Tensor, caption_tokens: torch.Tensor):
        self.image_features = image_features  # [N, seq_len, feature_dim]
        self.caption_tokens = caption_tokens  # [N, max_len]

    def __len__(self):
        return len(self.caption_tokens)

    def __getitem__(self, idx):
        return {
            'image_features': self.image_features[idx],
            'caption_tokens': self.caption_tokens[idx],
        }


class CaptionTrainer:
    def __init__(self, config: CaptionTrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_model(self):
        """Initialize decoder (features are pre-extracted, no encoder to train)."""
        print("=" * 60)
        print("Setting up captioning model")
        print("=" * 60)

        # Load data config to get feature dimensions
        config_path = os.path.join(self.config.data_dir, "config.json")
        with open(config_path) as f:
            data_config = json.load(f)

        feature_dim = data_config['feature_dim']
        print(f"\nUsing pre-extracted CLIP features: d_model={feature_dim}")

        # Create decoder with cross-attention
        base_config = MODEL_CONFIGS[self.config.decoder_config]
        decoder_config = ModelConfig(
            d_model=feature_dim,  # Match CLIP feature dim
            n_heads=base_config.n_heads,
            n_layers=self.config.decoder_n_layers,
            d_ff=base_config.d_ff,
            dropout=base_config.dropout,
            vocab_size=base_config.vocab_size,
            max_seq_len=self.config.max_caption_len,
            has_cross_attention=True,  # Enable cross-attention
        )

        self.decoder = DiffusionTransformer(decoder_config)
        self.decoder.to(self.device)

        print(f"Decoder: {decoder_config.n_layers} layers, d_model={decoder_config.d_model}")
        print(f"Cross-attention: Enabled")

        # Count parameters
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"\nTrainable parameters: {decoder_params:,}")
        print("=" * 60)

        # Create diffusion process
        self.diffusion = DiscreteDiffusion(
            vocab_size=decoder_config.vocab_size,
            mask_token_id=3,
            pad_token_id=0,
            schedule="cosine",
        )

        self.decoder_config = decoder_config

    def _setup_data(self):
        """Load pre-extracted image features and captions."""
        print("\nLoading data...")

        # Load training data
        train_features = torch.load(os.path.join(self.config.data_dir, "train_image_features.pt"))
        train_captions = torch.load(os.path.join(self.config.data_dir, "train_captions.pt"))

        # Load validation data
        val_features = torch.load(os.path.join(self.config.data_dir, "val_image_features.pt"))
        val_captions = torch.load(os.path.join(self.config.data_dir, "val_captions.pt"))

        print(f"  Train: {len(train_captions):,} examples")
        print(f"  Val: {len(val_captions):,} examples")
        print(f"  Image features: {train_features.shape[1:]} (seq_len, feature_dim)")
        print(f"  Caption length: {train_captions.shape[1]}")

        # Create datasets
        train_dataset = CaptionDataset(train_features, train_captions)
        val_dataset = CaptionDataset(val_features, val_captions)

        # Create dataloaders
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
        """Setup optimizer and gradient scaler."""
        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp)

    def get_lr(self, step: int) -> float:
        """Learning rate schedule with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        return self.config.learning_rate

    def train_step(self, batch):
        """Single training step."""
        self.decoder.train()

        image_features = batch['image_features'].to(self.device)  # [B, seq_len, d_model]
        captions = batch['caption_tokens'].to(self.device)  # [B, max_len]

        # Create attention masks
        image_mask = torch.ones(image_features.shape[:2], device=self.device)  # All valid
        caption_mask = (captions != 0).float()  # 1 for non-pad

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            loss, metrics = self.diffusion.training_losses(
                self.decoder,
                captions,
                attention_mask=caption_mask,
                encoder_output=image_features,
                encoder_attention_mask=image_mask,
            )

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return metrics

    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.decoder.eval()

        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch in self.val_loader:
            image_features = batch['image_features'].to(self.device)
            captions = batch['caption_tokens'].to(self.device)

            image_mask = torch.ones(image_features.shape[:2], device=self.device)
            caption_mask = (captions != 0).float()

            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                loss, metrics = self.diffusion.training_losses(
                    self.decoder,
                    captions,
                    attention_mask=caption_mask,
                    encoder_output=image_features,
                    encoder_attention_mask=image_mask,
                )

            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': step,
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_config': self.decoder_config.__dict__,
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
        print("Starting Captioning Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print("=" * 60)

        pbar = tqdm(total=self.config.max_steps, desc="Training")

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Update learning rate
                lr = self.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Training step
                metrics = self.train_step(batch)

                # Log
                if self.global_step % self.config.log_every == 0:
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'acc': f"{metrics['accuracy']:.3f}",
                        'lr': f"{lr:.2e}",
                    })
                    print(f"Step {self.global_step}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.3f}, lr={lr:.2e}")

                # Validation
                if self.global_step % self.config.eval_every == 0:
                    val_loss, val_acc = self.validate()
                    print(f"\n[Val] loss={val_loss:.4f}, acc={val_acc:.3f}")

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        print("  New best model!")

                    # Save checkpoint
                    if self.global_step > 0 and self.global_step % self.config.save_every == 0:
                        self.save_checkpoint(self.global_step, is_best)

                self.global_step += 1
                pbar.update(1)

                if self.global_step >= self.config.max_steps:
                    break

        pbar.close()

        # Final save
        self.save_checkpoint(self.global_step)
        final_path = os.path.join(self.config.checkpoint_dir, "final.pt")
        import shutil
        shutil.copy(
            os.path.join(self.config.checkpoint_dir, f"step_{self.global_step}.pt"),
            final_path
        )
        print(f"\nSaved final checkpoint: {final_path}")

        print("\n" + "=" * 60)
        print("Captioning Training Complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=str, default="data_captions_synthetic")

    # Model
    parser.add_argument("--decoder_config", type=str, default="small")
    parser.add_argument("--decoder_n_layers", type=int, default=6)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_caption")
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    config = CaptionTrainConfig(
        data_dir=args.data_dir,
        decoder_config=args.decoder_config,
        decoder_n_layers=args.decoder_n_layers,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
    )

    trainer = CaptionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
