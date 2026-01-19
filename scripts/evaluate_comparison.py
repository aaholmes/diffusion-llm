#!/usr/bin/env python3
"""
Evaluate and compare AR, Diffusion, and SDD models.
Computes perplexity, accuracy, and generates samples.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_ar_model(checkpoint_path, device):
    """Load autoregressive model."""
    from src.training.train_ar_text import CausalTransformer, ARConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ARConfig(**checkpoint['config'])
    model = CausalTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def load_diffusion_model(checkpoint_path, device):
    """Load discrete diffusion model."""
    from src.core.model import DiffusionTransformer, ModelConfig
    from src.core.diffusion import DiscreteDiffusion

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle different checkpoint formats
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif 'config' in checkpoint:
        config = ModelConfig(**checkpoint['config'])
    else:
        raise KeyError(f"No config found in checkpoint. Keys: {checkpoint.keys()}")

    if not isinstance(config, ModelConfig):
        config = ModelConfig(**config) if isinstance(config, dict) else config

    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    diffusion = DiscreteDiffusion(
        vocab_size=config.vocab_size,
        mask_token_id=3,
    )
    return model, config, diffusion


def load_sdd_model(checkpoint_path, device):
    """Load Sparse Distribution Diffusion model."""
    from src.core.sparse_diffusion import SparseDiffusion, SDDConfig as DiffusionConfig
    from src.core.sparse_model import BilateralSparseDenoiser, SparseModelConfig
    import torch.nn as nn

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config') or checkpoint.get('config')
    sdd_cfg = checkpoint.get('sdd_config', {})

    # Create model config
    model_config = SparseModelConfig(
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config.get('d_ff', config['d_model'] * 4),
        embed_dim=config.get('embed_dim', 64),
        k=config.get('k', 8),
        encoder_dim=config.get('encoder_dim', None),  # None = no cross-attention
    )

    model = BilateralSparseDenoiser(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create diffusion config and object
    embed_dim = config.get('embed_dim', 64)
    embedding_table = nn.Embedding(config['vocab_size'], embed_dim).to(device)
    # Copy embeddings from model if available
    if hasattr(model, 'token_embedding'):
        embedding_table.weight.data = model.token_embedding.weight.data.clone()

    diffusion_config = DiffusionConfig(
        vocab_size=config['vocab_size'],
        embed_dim=embed_dim,
        k=config.get('k', 8),
    )
    diffusion = SparseDiffusion(diffusion_config, embedding_table)

    return model, config, diffusion


@torch.no_grad()
def evaluate_ar(model, val_loader, device, vocab_size, max_batches=100):
    """Evaluate AR model - compute cross-entropy loss and accuracy."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for i, (batch,) in enumerate(tqdm(val_loader, desc="Eval AR", total=max_batches)):
        if i >= max_batches:
            break

        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=0,
            reduction='sum'
        )

        mask = targets != 0
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == targets[mask]).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += mask.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
    }


@torch.no_grad()
def evaluate_diffusion(model, diffusion, val_loader, device, max_batches=100):
    """Evaluate diffusion model - compute loss at various noise levels."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0

    for i, (batch,) in enumerate(tqdm(val_loader, desc="Eval Diffusion", total=max_batches)):
        if i >= max_batches:
            break

        batch = batch.to(device)
        B = batch.shape[0]

        # Sample random timesteps
        t = torch.rand(B, device=device)

        # Add noise
        noisy, mask = diffusion.q_sample(batch, t)

        # Predict
        logits = model(noisy, t)

        # Loss only on masked positions
        loss = F.cross_entropy(
            logits.reshape(-1, diffusion.vocab_size),
            batch.reshape(-1),
            ignore_index=0,
            reduction='none'
        ).reshape(B, -1)

        masked_loss = (loss * mask).sum()

        # Accuracy on masked positions
        preds = logits.argmax(dim=-1)
        correct = ((preds == batch) & mask).sum().item()

        total_loss += masked_loss.item()
        total_correct += correct
        total_masked += mask.sum().item()

    avg_loss = total_loss / total_masked
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_masked

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'masked_accuracy': accuracy,
    }


@torch.no_grad()
def evaluate_sdd(model, diffusion, val_loader, device, vocab_size, max_batches=100):
    """Evaluate SDD model - compute loss on sparse predictions."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_positions = 0

    for i, (batch,) in enumerate(tqdm(val_loader, desc="Eval SDD", total=max_batches)):
        if i >= max_batches:
            break

        batch = batch.to(device)
        B, L = batch.shape

        # Create sparse state from tokens and add noise
        state = diffusion.state_from_tokens(batch)
        t = torch.rand(B, device=device) * 0.8 + 0.1  # Avoid extremes
        noisy_state = diffusion.add_noise(state, t)

        # Forward pass - model predicts logits over full vocabulary
        logits = model(noisy_state, t)  # [B, L, vocab_size]

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            batch.reshape(-1),
            ignore_index=0,
            reduction='none'
        ).reshape(B, -1)

        valid_mask = batch != 0
        total_loss += (loss * valid_mask).sum().item()
        total_positions += valid_mask.sum().item()

        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = ((preds == batch) & valid_mask).sum().item()
        total_correct += correct

    avg_loss = total_loss / total_positions if total_positions > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_positions if total_positions > 0 else 0

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
    }


@torch.no_grad()
def generate_samples_ar(model, tokenizer, device, num_samples=3, max_len=64):
    """Generate samples from AR model."""
    samples = []
    for _ in range(num_samples):
        tokens = model.generate(max_len=max_len, temperature=0.8, device=device)
        text = tokenizer.decode(tokens[0].tolist())
        samples.append(text)
    return samples


@torch.no_grad()
def generate_samples_diffusion(model, diffusion, tokenizer, device, num_samples=3, seq_len=64, num_steps=50):
    """Generate samples from diffusion model."""
    samples = []
    for _ in range(num_samples):
        tokens = diffusion.sample(model, batch_size=1, seq_len=seq_len, num_steps=num_steps, device=device)
        text = tokenizer.decode(tokens[0].tolist())
        samples.append(text)
    return samples


@torch.no_grad()
def generate_samples_sdd(model, diffusion, tokenizer, device, num_samples=3, seq_len=64, num_steps=50):
    """Generate samples from SDD model."""
    samples = []
    for _ in range(num_samples):
        tokens, _ = diffusion.sample(model, batch_size=1, seq_len=seq_len, num_steps=num_steps, device=device)
        text = tokenizer.decode(tokens[0].tolist())
        samples.append(text)
    return samples


def benchmark_inference(generate_fn, num_runs=5):
    """Benchmark inference speed."""
    # Warmup
    generate_fn()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(num_runs):
        generate_fn()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    return elapsed / num_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_full')
    parser.add_argument('--ar_checkpoint', type=str, default='checkpoints_compare_ar/best.pt')
    parser.add_argument('--diffusion_checkpoint', type=str, default='checkpoints_compare_diffusion/best.pt')
    parser.add_argument('--sdd_checkpoint', type=str, default='checkpoints_compare_sdd/best.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_batches', type=int, default=100)
    parser.add_argument('--generate_samples', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load validation data
    print(f"\nLoading validation data from {args.data_dir}...")
    val_tokens = torch.load(os.path.join(args.data_dir, 'val_tokens.pt'))
    val_dataset = TensorDataset(val_tokens)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Val samples: {len(val_tokens)}")

    # Load tokenizer for generation
    tokenizer = None
    tokenizer_path = os.path.join(args.data_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)

    results = {}

    # Evaluate AR model
    if os.path.exists(args.ar_checkpoint):
        print(f"\n{'='*50}")
        print("Evaluating Autoregressive Model")
        print('='*50)
        model, config = load_ar_model(args.ar_checkpoint, device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        metrics = evaluate_ar(model, val_loader, device, config.vocab_size, args.max_batches)
        results['AR'] = metrics
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")

        if args.generate_samples and tokenizer:
            print("\n  Generated Samples:")
            samples = generate_samples_ar(model, tokenizer, device)
            for i, s in enumerate(samples):
                print(f"    [{i+1}] {s[:200]}...")

        if args.benchmark:
            gen_time = benchmark_inference(lambda: model.generate(max_len=64, device=device))
            print(f"\n  Generation time (64 tokens): {gen_time*1000:.1f}ms")
            results['AR']['gen_time_ms'] = gen_time * 1000
    else:
        print(f"\nAR checkpoint not found: {args.ar_checkpoint}")

    # Evaluate Diffusion model
    if os.path.exists(args.diffusion_checkpoint):
        print(f"\n{'='*50}")
        print("Evaluating Discrete Diffusion Model")
        print('='*50)
        model, config, diffusion = load_diffusion_model(args.diffusion_checkpoint, device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        metrics = evaluate_diffusion(model, diffusion, val_loader, device, args.max_batches)
        results['Diffusion'] = metrics
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Masked Accuracy: {metrics['masked_accuracy']:.3f}")

        if args.generate_samples and tokenizer:
            print("\n  Generated Samples:")
            samples = generate_samples_diffusion(model, diffusion, tokenizer, device)
            for i, s in enumerate(samples):
                print(f"    [{i+1}] {s[:200]}...")

        if args.benchmark:
            gen_time = benchmark_inference(
                lambda: diffusion.sample(model, batch_size=1, seq_len=64, num_steps=50, device=device)
            )
            print(f"\n  Generation time (64 tokens, 50 steps): {gen_time*1000:.1f}ms")
            results['Diffusion']['gen_time_ms'] = gen_time * 1000
    else:
        print(f"\nDiffusion checkpoint not found: {args.diffusion_checkpoint}")

    # Evaluate SDD model
    if os.path.exists(args.sdd_checkpoint):
        print(f"\n{'='*50}")
        print("Evaluating Sparse Distribution Diffusion Model")
        print('='*50)
        model, config, diffusion = load_sdd_model(args.sdd_checkpoint, device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        metrics = evaluate_sdd(model, diffusion, val_loader, device, vocab_size=config['vocab_size'], max_batches=args.max_batches)
        results['SDD'] = metrics
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")

        if args.generate_samples and tokenizer:
            print("\n  Generated Samples:")
            samples = generate_samples_sdd(model, diffusion, tokenizer, device)
            for i, s in enumerate(samples):
                print(f"    [{i+1}] {s[:200]}...")

        if args.benchmark:
            gen_time = benchmark_inference(
                lambda: diffusion.sample(model, batch_size=1, seq_len=64, num_steps=50, device=device)
            )
            print(f"\n  Generation time (64 tokens, 50 steps): {gen_time*1000:.1f}ms")
            results['SDD']['gen_time_ms'] = gen_time * 1000
    else:
        print(f"\nSDD checkpoint not found: {args.sdd_checkpoint}")

    # Summary table
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print('='*50)
    print(f"{'Model':<15} {'Perplexity':>12} {'Accuracy':>12} {'Loss':>10}")
    print('-'*50)
    for name, metrics in results.items():
        acc_key = 'accuracy' if 'accuracy' in metrics else 'masked_accuracy'
        print(f"{name:<15} {metrics['perplexity']:>12.2f} {metrics[acc_key]:>12.3f} {metrics['loss']:>10.4f}")

    if args.benchmark and results:
        print(f"\n{'Model':<15} {'Gen Time (ms)':>15}")
        print('-'*30)
        for name, metrics in results.items():
            if 'gen_time_ms' in metrics:
                print(f"{name:<15} {metrics['gen_time_ms']:>15.1f}")

    # Save results
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to comparison_results.json")


if __name__ == '__main__':
    main()
