#!/usr/bin/env python3
"""
Simple autoregressive language model trainer for text.
Comparable baseline for diffusion models.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class ARConfig:
    vocab_size: int = 8192
    max_seq_len: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1


class CausalTransformer(nn.Module):
    """Standard causal (autoregressive) transformer."""

    def __init__(self, config: ARConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x: [batch, seq_len] token ids
        returns: [batch, seq_len, vocab_size] logits
        """
        B, L = x.shape

        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.token_embedding(x) + self.position_embedding(positions)
        h = self.dropout(h)

        # Causal attention mask
        mask = self.causal_mask[:L, :L]

        h = self.transformer(h, mask=mask, is_causal=True)
        logits = self.output_proj(h)

        return logits

    @torch.no_grad()
    def generate(self, prompt=None, max_len=128, temperature=0.8, top_k=50, device='cuda'):
        """Generate text autoregressively."""
        self.eval()

        if prompt is None:
            tokens = torch.tensor([[1]], device=device)  # BOS
        else:
            tokens = prompt.clone()

        for _ in range(max_len - tokens.shape[1]):
            logits = self(tokens)[:, -1, :]  # Last position
            logits = logits / temperature

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == 2:  # EOS
                break

        return tokens


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_tokens = torch.load(os.path.join(args.data_dir, 'train_tokens.pt'))
    val_tokens = torch.load(os.path.join(args.data_dir, 'val_tokens.pt'))

    with open(os.path.join(args.data_dir, 'config.json')) as f:
        data_config = json.load(f)

    print(f"  Train: {train_tokens.shape}")
    print(f"  Val: {val_tokens.shape}")

    # Create model
    config = ARConfig(
        vocab_size=data_config['vocab_size'],
        max_seq_len=data_config['max_seq_len'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )

    model = CausalTransformer(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Data loaders
    train_dataset = TensorDataset(train_tokens)
    val_dataset = TensorDataset(val_tokens)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # LR schedule with warmup
    def get_lr(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Training
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    step = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)

    pbar = tqdm(total=args.max_steps, desc="Training")

    while step < args.max_steps:
        model.train()

        try:
            (batch,) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            (batch,) = next(train_iter)

        batch = batch.to(device)

        # Input: all but last token, Target: all but first token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
                ignore_index=0  # Ignore padding
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1

        # Logging
        if step % args.log_every == 0:
            # Calculate accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = targets != 0
                acc = (preds[mask] == targets[mask]).float().mean().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        pbar.update(1)

        # Validation
        if step % args.eval_every == 0:
            model.eval()
            val_losses = []
            val_accs = []

            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]

                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        logits = model(inputs)
                        loss = F.cross_entropy(
                            logits.reshape(-1, config.vocab_size),
                            targets.reshape(-1),
                            ignore_index=0
                        )

                    val_losses.append(loss.item())

                    preds = logits.argmax(dim=-1)
                    mask = targets != 0
                    acc = (preds[mask] == targets[mask]).float().mean().item()
                    val_accs.append(acc)

                    if len(val_losses) >= 100:  # Sample of val set
                        break

            val_loss = sum(val_losses) / len(val_losses)
            val_acc = sum(val_accs) / len(val_accs)
            val_ppl = torch.exp(torch.tensor(val_loss)).item()

            print(f"\n[Val] loss={val_loss:.4f}, ppl={val_ppl:.2f}, acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'step': step,
                    'val_loss': val_loss,
                }, os.path.join(args.checkpoint_dir, 'best.pt'))
                print("  New best model!")

        # Save checkpoint
        if step % args.save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'step': step,
            }, os.path.join(args.checkpoint_dir, f'step_{step}.pt'))

        # Generate samples
        if step % args.generate_every == 0:
            model.eval()
            print("\n--- Generated Samples ---")

            from tokenizers import Tokenizer
            tokenizer_path = os.path.join(args.data_dir, 'tokenizer.json')
            if os.path.exists(tokenizer_path):
                tokenizer = Tokenizer.from_file(tokenizer_path)

                for i in range(3):
                    tokens = model.generate(max_len=64, temperature=0.8, device=device)
                    text = tokenizer.decode(tokens[0].tolist())
                    print(f"[{i+1}] {text}")

            print("-" * 25 + "\n")

    pbar.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_full')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_ar_text')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--d_model', type=int, default=384)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--generate_every', type=int, default=5000)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
