#!/usr/bin/env python3
"""
Evaluate trained diffusion language model.

Metrics:
- Perplexity: How surprised is the model by held-out text? Lower = better.
- Self-BLEU: Diversity of generations. Lower = more diverse.
- BLEU vs reference: How similar are generations to real TinyStories?

Usage:
    python evaluate.py                              # Full evaluation
    python evaluate.py --perplexity_only            # Just perplexity
    python evaluate.py --num_samples 100            # More samples for BLEU
"""

import argparse
import math
from collections import Counter
from typing import List

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from src.core.model import DiffusionTransformer, create_model
from src.core.diffusion import DiscreteDiffusion


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint.get("model_config", "small")

    if isinstance(model_config, str):
        model = create_model(model_config, vocab_size=8192, max_seq_len=256)
    else:
        model = DiffusionTransformer(model_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_perplexity(
    model: DiffusionTransformer,
    val_data_path: str,
    num_batches: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Compute perplexity on validation data.

    Perplexity = exp(average cross-entropy loss)
    Lower = better (model is less "surprised" by the data)
    """
    device = next(model.parameters()).device
    diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

    # Load validation data
    val_tokens = torch.load(val_data_path, weights_only=True)

    total_loss = 0.0
    total_tokens = 0
    num_samples = 0

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing perplexity"):
            # Sample random batch
            indices = torch.randint(0, len(val_tokens), (batch_size,))
            batch = val_tokens[indices].to(device)

            # Compute loss at random timestep (standard diffusion training loss)
            loss, metrics = diffusion.training_losses(model, batch)

            # Accumulate
            total_loss += metrics["loss"] * batch_size
            num_samples += batch_size

    avg_loss = total_loss / num_samples
    perplexity = math.exp(avg_loss)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": num_samples,
    }


def get_ngrams(tokens: List[int], n: int) -> List[tuple]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(candidate: List[int], references: List[List[int]], max_n: int = 4) -> float:
    """
    Compute BLEU score for a single candidate against references.

    BLEU measures n-gram precision with brevity penalty.
    """
    # Collect reference n-gram counts
    ref_ngram_counts = [Counter() for _ in range(max_n)]
    ref_lengths = [len(ref) for ref in references]

    for ref in references:
        for n in range(1, max_n + 1):
            ngrams = get_ngrams(ref, n)
            ref_ngram_counts[n-1].update(ngrams)

    # Count candidate n-grams
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue

        cand_counts = Counter(cand_ngrams)

        # Clipped counts (can't exceed reference count)
        clipped = sum(
            min(count, ref_ngram_counts[n-1].get(ng, 0))
            for ng, count in cand_counts.items()
        )

        precision = clipped / len(cand_ngrams)
        precisions.append(precision)

    # Geometric mean of precisions
    if 0 in precisions:
        return 0.0

    log_precision = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    cand_len = len(candidate)
    closest_ref_len = min(ref_lengths, key=lambda r: abs(r - cand_len))

    if cand_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / cand_len)

    bleu = bp * math.exp(log_precision)
    return bleu


def compute_self_bleu(samples: List[List[int]], max_n: int = 4) -> float:
    """
    Compute Self-BLEU: average BLEU of each sample against all others.

    Lower = more diverse generations.
    """
    if len(samples) < 2:
        return 0.0

    total_bleu = 0.0
    for i, sample in enumerate(samples):
        references = [s for j, s in enumerate(samples) if j != i]
        total_bleu += compute_bleu(sample, references, max_n)

    return total_bleu / len(samples)


def generate_samples(
    model: DiffusionTransformer,
    diffusion: DiscreteDiffusion,
    num_samples: int,
    seq_len: int = 128,
    steps: int = 100,
    temperature: float = 0.8,
    device: str = "cpu",
) -> List[List[int]]:
    """Generate samples and return as token lists."""
    samples = []
    batch_size = min(num_samples, 10)  # Generate in batches

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
        n = min(batch_size, num_samples - i)
        batch = diffusion.sample(
            model, batch_size=n, seq_len=seq_len,
            num_steps=steps, temperature=temperature, device=device
        )

        for sample in batch:
            tokens = sample.tolist()
            # Remove special tokens
            try:
                eos_idx = tokens.index(2)
                tokens = tokens[1:eos_idx]
            except ValueError:
                tokens = tokens[1:]
            samples.append(tokens)

    return samples


def evaluate_generation_quality(
    model: DiffusionTransformer,
    val_data_path: str,
    num_samples: int = 50,
    steps: int = 100,
    temperature: float = 0.8,
) -> dict:
    """
    Evaluate generation quality using BLEU metrics.

    - Self-BLEU: Diversity (lower = better)
    - Ref-BLEU: Similarity to real TinyStories (higher = better)
    """
    device = next(model.parameters()).device
    diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    generated = generate_samples(
        model, diffusion, num_samples,
        steps=steps, temperature=temperature, device=device
    )

    # Load reference samples from validation set
    val_tokens = torch.load(val_data_path, weights_only=True)
    ref_indices = torch.randint(0, len(val_tokens), (num_samples,))
    references = []
    for idx in ref_indices:
        tokens = val_tokens[idx].tolist()
        # Remove padding and special tokens
        try:
            eos_idx = tokens.index(2)
            tokens = tokens[1:eos_idx]
        except ValueError:
            tokens = [t for t in tokens[1:] if t != 0]
        references.append(tokens)

    # Self-BLEU (diversity)
    print("Computing Self-BLEU (diversity)...")
    self_bleu = compute_self_bleu(generated[:min(50, len(generated))])

    # Reference BLEU (similarity to real stories)
    print("Computing Ref-BLEU (similarity to real stories)...")
    ref_bleu_scores = []
    for gen in generated:
        # Compare each generation to all references
        bleu = compute_bleu(gen, references)
        ref_bleu_scores.append(bleu)

    avg_ref_bleu = sum(ref_bleu_scores) / len(ref_bleu_scores)

    # Average generation length
    avg_len = sum(len(g) for g in generated) / len(generated)
    avg_ref_len = sum(len(r) for r in references) / len(references)

    return {
        "self_bleu": self_bleu,
        "ref_bleu": avg_ref_bleu,
        "avg_gen_length": avg_len,
        "avg_ref_length": avg_ref_len,
        "num_samples": num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion LM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_long/final.pt")
    parser.add_argument("--val_data", type=str, default="data_full/val_tokens.pt")
    parser.add_argument("--perplexity_only", action="store_true")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # Perplexity
    print("\n" + "="*60)
    print("Perplexity Evaluation")
    print("="*60)
    ppl_results = compute_perplexity(model, args.val_data)
    print(f"\nResults:")
    print(f"  Average loss: {ppl_results['avg_loss']:.4f}")
    print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
    print(f"  (Lower is better. GPT-2 small on similar data: ~15-25)")

    if args.perplexity_only:
        return

    # Generation quality
    print("\n" + "="*60)
    print("Generation Quality Evaluation")
    print("="*60)
    gen_results = evaluate_generation_quality(
        model, args.val_data,
        num_samples=args.num_samples,
        steps=args.steps,
        temperature=args.temperature,
    )

    print(f"\nResults:")
    print(f"  Self-BLEU: {gen_results['self_bleu']:.4f}")
    print(f"    (Lower = more diverse. Typical range: 0.3-0.8)")
    print(f"  Ref-BLEU: {gen_results['ref_bleu']:.4f}")
    print(f"    (Higher = more similar to real stories. Typical: 0.01-0.1)")
    print(f"  Avg generation length: {gen_results['avg_gen_length']:.1f} tokens")
    print(f"  Avg reference length: {gen_results['avg_ref_length']:.1f} tokens")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
    print(f"  Self-BLEU (diversity): {gen_results['self_bleu']:.4f}")
    print(f"  Ref-BLEU (quality): {gen_results['ref_bleu']:.4f}")


if __name__ == "__main__":
    main()
