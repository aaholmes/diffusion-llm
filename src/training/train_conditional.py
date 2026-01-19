#!/usr/bin/env python3
"""
Training script for conditional diffusion language model (Stage 2).

This implements staged training:
1. (Already done) Train unconditional denoiser on full data
2. (This script) Freeze denoiser, add encoder + cross-attention, train on conditional pairs

The key insight: The unconditional model has no cross-attention. We create a new
decoder with cross-attention enabled, load the compatible weights (self-attention,
FFN, embeddings), and train only the new encoder + cross-attention layers.

Usage:
    python train_conditional.py --denoiser checkpoints_long/final.pt --max_steps 5000

    # Quick proof of concept
    python train_conditional.py --denoiser checkpoints_long/final.pt --max_steps 1000 --batch_size 32
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.core.model import (
    DiffusionTransformer, TextEncoder, ModelConfig, MODEL_CONFIGS,
    create_model, create_encoder
)
from src.core.diffusion import DiscreteDiffusion


@dataclass
class ConditionalTrainConfig:
    """Configuration for conditional training."""
    # Pretrained denoiser
    denoiser_checkpoint: str = "checkpoints_long/final.pt"

    # Resume from checkpoint
    resume_checkpoint: str = None  # Path to checkpoint to resume from

    # Encoder config (use same architecture as decoder for simplicity)
    encoder_config: str = "small"
    encoder_n_layers: int = 4  # Smaller encoder is fine

    # Data
    data_dir: str = "data_conditional"
    tokenizer_path: str = "data_full/tokenizer.json"
    max_encoder_len: int = 64
    max_decoder_len: int = 192

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 10000

    # Optimizer
    learning_rate: float = 1e-4  # Lower LR for stage 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LR Schedule
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1

    # Validation & Checkpointing
    eval_every: int = 250
    save_every: int = 1000
    num_eval_batches: int = 50
    checkpoint_dir: str = "checkpoints_conditional"

    # Logging
    log_every: int = 10
    use_wandb: bool = False

    # Hardware
    device: str = "cuda"
    use_amp: bool = True
    num_workers: int = 4

    # Special tokens
    mask_token_id: int = 3
    pad_token_id: int = 0
    vocab_size: int = 8192


class ConditionalTrainer:
    """Trainer for conditional diffusion model (Stage 2)."""

    def __init__(self, config: ConditionalTrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if provided
        if config.resume_checkpoint:
            self._load_checkpoint(config.resume_checkpoint)

    def _setup_model(self):
        """Initialize encoder and decoder with pretrained weights."""
        print("=" * 60)
        print("Setting up conditional model")
        print("=" * 60)

        # Load pretrained denoiser checkpoint
        print(f"\nLoading pretrained denoiser: {self.config.denoiser_checkpoint}")
        checkpoint = torch.load(
            self.config.denoiser_checkpoint,
            map_location='cpu',
            weights_only=False
        )

        # Get the original config
        orig_config = checkpoint['model_config']
        if isinstance(orig_config, dict):
            orig_config = ModelConfig(**orig_config)

        print(f"  Original config: d_model={orig_config.d_model}, n_layers={orig_config.n_layers}")
        print(f"  Original has_cross_attention: {orig_config.has_cross_attention}")

        # Create new decoder config WITH cross-attention
        decoder_config = ModelConfig(
            d_model=orig_config.d_model,
            n_heads=orig_config.n_heads,
            n_layers=orig_config.n_layers,
            d_ff=orig_config.d_ff,
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_decoder_len,
            dropout=orig_config.dropout,
            pad_token_id=self.config.pad_token_id,
            has_cross_attention=True,  # Enable cross-attention!
        )

        # Create new decoder with cross-attention
        print("\nCreating decoder with cross-attention...")
        self.decoder = DiffusionTransformer(decoder_config)

        # Load compatible weights from pretrained model
        pretrained_state = checkpoint['model_state_dict']
        new_state = self.decoder.state_dict()

        # Find which weights can be loaded
        loaded_keys = []
        new_keys = []
        for key in new_state:
            if key in pretrained_state and pretrained_state[key].shape == new_state[key].shape:
                new_state[key] = pretrained_state[key]
                loaded_keys.append(key)
            else:
                new_keys.append(key)

        self.decoder.load_state_dict(new_state)
        print(f"  Loaded {len(loaded_keys)} weight tensors from pretrained model")
        print(f"  New layers (cross-attention): {len(new_keys)}")

        # Zero-initialize cross-attention output projections
        # This ensures decoder behaves exactly like pretrained model at init
        # (cross-attention contributes zero, gradually learns to contribute)
        print("\n  Zero-initializing cross-attention output projections...")
        with torch.no_grad():
            for block in self.decoder.blocks:
                if block.has_cross_attention:
                    # Zero the output projection so cross-attn initially outputs zeros
                    nn.init.zeros_(block.cross_attn.out_proj.weight)
                    if block.cross_attn.out_proj.bias is not None:
                        nn.init.zeros_(block.cross_attn.out_proj.bias)
        print("  Cross-attention initialized to output zeros at start")

        # Create encoder
        print("\nCreating encoder...")
        encoder_config = ModelConfig(
            d_model=orig_config.d_model,  # Must match decoder
            n_heads=orig_config.n_heads,
            n_layers=self.config.encoder_n_layers,
            d_ff=orig_config.d_ff,
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_encoder_len,
            dropout=orig_config.dropout,
            pad_token_id=self.config.pad_token_id,
        )
        self.encoder = TextEncoder(encoder_config)

        # Freeze decoder except cross-attention
        print("\nFreezing decoder (keeping cross-attention trainable)...")
        self._freeze_decoder_except_cross_attention()

        # Count trainable parameters
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = enc_params + dec_params

        print(f"\nTrainable parameters:")
        print(f"  Encoder: {enc_params:,}")
        print(f"  Decoder (cross-attention only): {dec_params:,}")
        print(f"  Total: {total_params:,}")

        # Move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Create diffusion process
        self.diffusion = DiscreteDiffusion(
            vocab_size=self.config.vocab_size,
            mask_token_id=self.config.mask_token_id,
            pad_token_id=self.config.pad_token_id,
        )

        print("=" * 60)

    def _freeze_decoder_except_cross_attention(self):
        """Freeze all decoder parameters except cross-attention layers."""
        # First freeze everything
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Unfreeze cross-attention in each block
        for block in self.decoder.blocks:
            if block.has_cross_attention:
                for param in block.norm_cross.parameters():
                    param.requires_grad = True
                for param in block.cross_attn.parameters():
                    param.requires_grad = True
                # dropout_cross has no parameters

    def _setup_data(self):
        """Load conditional training data."""
        print("\nLoading conditional data...")

        data_dir = Path(self.config.data_dir)

        # Load encoder inputs (first sentences)
        train_enc = torch.load(data_dir / "train_encoder.pt", weights_only=True)
        val_enc = torch.load(data_dir / "val_encoder.pt", weights_only=True)

        # Load decoder targets (rest of story)
        train_dec = torch.load(data_dir / "train_decoder.pt", weights_only=True)
        val_dec = torch.load(data_dir / "val_decoder.pt", weights_only=True)

        print(f"  Train: {len(train_enc):,} pairs")
        print(f"  Val: {len(val_enc):,} pairs")
        print(f"  Encoder seq len: {train_enc.shape[1]}")
        print(f"  Decoder seq len: {train_dec.shape[1]}")

        # Create datasets
        self.train_dataset = TensorDataset(train_enc, train_dec)
        self.val_dataset = TensorDataset(val_enc, val_dec)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def _setup_optimizer(self):
        """Initialize optimizer for trainable parameters only."""
        # Collect trainable parameters
        trainable_params = list(self.encoder.parameters())
        trainable_params += [p for p in self.decoder.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp)

    def get_lr(self, step: int) -> float:
        """Learning rate with warmup and cosine decay."""
        max_lr = self.config.learning_rate
        min_lr = max_lr * self.config.min_lr_ratio
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps

        if step < warmup:
            return max_lr * step / warmup
        else:
            progress = (step - warmup) / (max_steps - warmup)
            progress = min(progress, 1.0)
            return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def train_step(self, enc_input: torch.Tensor, dec_target: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step."""
        self.encoder.train()
        self.decoder.train()

        enc_input = enc_input.to(self.device)
        dec_target = dec_target.to(self.device)

        batch_size = dec_target.shape[0]

        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            # Sample timestep
            t = torch.rand(batch_size, device=self.device)

            # Apply noise (masking) to decoder target
            noisy_target, mask_positions = self.diffusion.q_sample(dec_target, t)

            # Create attention masks (1 = attend, 0 = ignore padding)
            enc_attention_mask = (enc_input != self.config.pad_token_id).float()
            dec_attention_mask = (dec_target != self.config.pad_token_id).float()

            # Encode condition
            encoder_output = self.encoder(enc_input, enc_attention_mask)

            # Decode with cross-attention
            logits = self.decoder(
                noisy_target, t,
                attention_mask=dec_attention_mask,
                encoder_output=encoder_output,
                encoder_attention_mask=enc_attention_mask,
            )

            # Compute loss only on masked positions
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                dec_target.view(-1),
                ignore_index=self.config.pad_token_id,
                reduction='none',
            )
            loss = loss.view(batch_size, -1)

            # Weight by mask
            mask_float = mask_positions.float()
            loss = (loss * mask_float).sum() / (mask_float.sum() + 1e-8)

            # Accuracy on masked positions
            preds = logits.argmax(dim=-1)
            correct = (preds == dec_target) & mask_positions
            accuracy = correct.sum().float() / (mask_positions.sum() + 1e-8)

        # Backward
        scaled_loss = loss / self.config.grad_accum_steps
        self.scaler.scale(scaled_loss).backward()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "mask_rate": mask_positions.float().mean().item(),
        }

    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        self.scaler.unscale_(self.optimizer)

        # Clip gradients
        all_params = list(self.encoder.parameters())
        all_params += [p for p in self.decoder.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return grad_norm.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation."""
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        total_acc = 0
        num_batches = 0

        for enc_input, dec_target in self.val_loader:
            if num_batches >= self.config.num_eval_batches:
                break

            enc_input = enc_input.to(self.device)
            dec_target = dec_target.to(self.device)
            batch_size = dec_target.shape[0]

            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                t = torch.rand(batch_size, device=self.device)
                noisy_target, mask_positions = self.diffusion.q_sample(dec_target, t)

                enc_attention_mask = (enc_input != self.config.pad_token_id).float()
                dec_attention_mask = (dec_target != self.config.pad_token_id).float()

                encoder_output = self.encoder(enc_input, enc_attention_mask)
                logits = self.decoder(
                    noisy_target, t,
                    attention_mask=dec_attention_mask,
                    encoder_output=encoder_output,
                    encoder_attention_mask=enc_attention_mask,
                )

                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    dec_target.view(-1),
                    ignore_index=self.config.pad_token_id,
                    reduction='none',
                )
                loss = loss.view(batch_size, -1)
                mask_float = mask_positions.float()
                loss = (loss * mask_float).sum() / (mask_float.sum() + 1e-8)

                preds = logits.argmax(dim=-1)
                correct = (preds == dec_target) & mask_positions
                accuracy = correct.sum().float() / (mask_positions.sum() + 1e-8)

            total_loss += loss.item()
            total_acc += accuracy.item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_acc / num_batches,
        }

    def save_checkpoint(self, name: str = None, is_best: bool = False):
        """Save checkpoint."""
        if name is None:
            name = f"step_{self.global_step}"

        checkpoint = {
            "global_step": self.global_step,
            "encoder_state_dict": self.encoder.state_dict(),
            "encoder_config": self.encoder.config,
            "decoder_state_dict": self.decoder.state_dict(),
            "decoder_config": self.decoder.config,
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

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load model states
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # Load optimizer and scaler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"  Resuming from step {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def generate_sample(self, prompt_tokens: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generate a sample given prompt tokens."""
        self.encoder.eval()
        self.decoder.eval()

        prompt_tokens = prompt_tokens.to(self.device)
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        batch_size = prompt_tokens.shape[0]

        # Encode prompt
        enc_attention_mask = (prompt_tokens != self.config.pad_token_id).float()
        encoder_output = self.encoder(prompt_tokens, enc_attention_mask)

        # Start with all masks
        x = torch.full(
            (batch_size, self.config.max_decoder_len),
            self.config.mask_token_id,
            device=self.device,
        )

        # Iteratively denoise
        for step in range(steps):
            t = torch.tensor([1.0 - step / steps] * batch_size, device=self.device)

            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                logits = self.decoder(
                    x, t,
                    encoder_output=encoder_output,
                    encoder_attention_mask=enc_attention_mask,
                )

            # Sample from logits
            probs = torch.softmax(logits / 0.8, dim=-1)  # temperature=0.8
            sampled = torch.multinomial(probs.view(-1, self.config.vocab_size), 1)
            sampled = sampled.view(batch_size, -1)

            # Unmask based on timestep
            mask_rate = 1.0 - (step + 1) / steps
            num_to_keep_masked = int(mask_rate * self.config.max_decoder_len)

            # Determine which positions to unmask
            is_masked = (x == self.config.mask_token_id)

            # For each sequence, randomly keep some masked
            for b in range(batch_size):
                masked_indices = is_masked[b].nonzero().squeeze(-1)
                if len(masked_indices) > num_to_keep_masked:
                    # Unmask some positions
                    perm = torch.randperm(len(masked_indices), device=self.device)
                    to_unmask = masked_indices[perm[num_to_keep_masked:]]
                    x[b, to_unmask] = sampled[b, to_unmask]

        return x

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Conditional Training (Stage 2)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print("=" * 60 + "\n")

        self.optimizer.zero_grad()
        accum_metrics = {"loss": 0, "accuracy": 0, "mask_rate": 0}
        accum_count = 0

        pbar = tqdm(total=self.config.max_steps, desc="Training", unit="step")
        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            try:
                enc_input, dec_target = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                enc_input, dec_target = next(data_iter)

            # Update LR
            lr = self.get_lr(self.global_step)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # Train step
            metrics = self.train_step(enc_input, dec_target)

            for key in accum_metrics:
                accum_metrics[key] += metrics.get(key, 0)
            accum_count += 1

            # Optimizer step
            if accum_count >= self.config.grad_accum_steps:
                grad_norm = self.optimizer_step()
                self.global_step += 1
                pbar.update(1)

                # Log
                if self.global_step % self.config.log_every == 0:
                    avg_loss = accum_metrics["loss"] / accum_count
                    avg_acc = accum_metrics["accuracy"] / accum_count

                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}", lr=f"{lr:.2e}")
                    print(f"Step {self.global_step}: loss={avg_loss:.4f}, acc={avg_acc:.3f}, lr={lr:.2e}", flush=True)

                accum_metrics = {"loss": 0, "accuracy": 0, "mask_rate": 0}
                accum_count = 0

                # Validation
                if self.global_step % self.config.eval_every == 0:
                    val_metrics = self.evaluate()
                    print(f"\n[Val] loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_accuracy']:.3f}", flush=True)

                    is_best = val_metrics["val_loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["val_loss"]
                        print("  New best model!", flush=True)

                # Save
                if self.global_step % self.config.save_every == 0:
                    is_best = (self.global_step % self.config.eval_every == 0 and
                              val_metrics["val_loss"] <= self.best_val_loss)
                    self.save_checkpoint(is_best=is_best)

        pbar.close()
        self.save_checkpoint("final")
        print("\n" + "=" * 60)
        print("Conditional Training Complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train conditional diffusion LM (Stage 2)")

    # Model
    parser.add_argument("--denoiser", type=str, default="checkpoints_long/final.pt",
                       help="Path to pretrained denoiser checkpoint")
    parser.add_argument("--encoder_n_layers", type=int, default=4,
                       help="Number of encoder layers (default: 4)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    # Data
    parser.add_argument("--data_dir", type=str, default="data_conditional")
    parser.add_argument("--max_encoder_len", type=int, default=64)
    parser.add_argument("--max_decoder_len", type=int, default=192)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Checkpointing & Logging
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_conditional")

    # Hardware
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()

    config = ConditionalTrainConfig(
        denoiser_checkpoint=args.denoiser,
        encoder_n_layers=args.encoder_n_layers,
        resume_checkpoint=args.resume,
        data_dir=args.data_dir,
        max_encoder_len=args.max_encoder_len,
        max_decoder_len=args.max_decoder_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
    )

    trainer = ConditionalTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
