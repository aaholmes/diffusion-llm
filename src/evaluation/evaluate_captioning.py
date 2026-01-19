#!/usr/bin/env python3
"""
Evaluate image captioning model on COCO validation set.

Usage:
    python evaluate_captioning.py --checkpoint checkpoints_caption_coco/final.pt --data_dir data_coco

Metrics:
    - CIDEr (primary metric for COCO captioning)
    - BLEU (1-4)
    - METEOR
    - ROUGE-L
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.model import DiffusionTransformer, ModelConfig
from src.core.diffusion import DiscreteDiffusion
from src.training.train_captioning import CaptionDataset


def load_tokenizer(tokenizer_path: str):
    """Load BPE tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tokenizer_path)


def decode_tokens(token_ids, tokenizer, skip_special_tokens=True):
    """Decode token IDs to text."""
    # Filter out special tokens if requested
    if skip_special_tokens:
        special_ids = {0, 1, 2, 3, 4}  # PAD, BOS, EOS, MASK, UNK
        token_ids = [t for t in token_ids if t not in special_ids]

    # Decode
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return text.strip()


@torch.no_grad()
def generate_captions(
    model,
    diffusion,
    image_features,
    tokenizer,
    num_steps=25,
    temperature=1.0,
    device='cuda'
):
    """Generate captions for a batch of images."""
    model.eval()

    batch_size = image_features.shape[0]
    max_len = model.config.max_seq_len

    # Create attention masks
    image_mask = torch.ones(image_features.shape[:2], device=device)

    # Generate using diffusion sampling
    generated = diffusion.sample(
        model,
        batch_size=batch_size,
        seq_len=max_len,
        num_steps=num_steps,
        temperature=temperature,
        device=device,
        encoder_output=image_features,
        encoder_attention_mask=image_mask,
    )

    # Decode to text
    captions = []
    for seq in generated:
        tokens = seq.cpu().tolist()
        text = decode_tokens(tokens, tokenizer)
        captions.append(text)

    return captions


def evaluate_coco_metrics(generated_captions, reference_captions_list):
    """
    Evaluate using COCO metrics.

    Args:
        generated_captions: Dict[int, str] - {image_id: generated_caption}
        reference_captions_list: Dict[int, List[str]] - {image_id: [ref1, ref2, ...]}

    Returns:
        Dict of metric scores
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        print("\nWARNING: pycocoevalcap not installed. Install with:")
        print("  pip install pycocoevalcap")
        return {}

    # Format: {image_id: [caption]}
    gts = {img_id: refs for img_id, refs in reference_captions_list.items()}
    res = {img_id: [cap] for img_id, cap in generated_captions.items()}

    # Compute metrics
    metrics = {}

    # CIDEr (primary metric)
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    metrics['CIDEr'] = cider_score * 100  # Scale to 0-100

    # BLEU
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    for i, score in enumerate(bleu_scores, 1):
        metrics[f'BLEU-{i}'] = score * 100

    # METEOR
    try:
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
        metrics['METEOR'] = meteor_score * 100
    except Exception as e:
        print(f"METEOR computation failed: {e}")

    # ROUGE-L
    try:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        metrics['ROUGE-L'] = rouge_score * 100
    except Exception as e:
        print(f"ROUGE-L computation failed: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data_coco", help="Data directory")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--max_samples", type=int, default=None, help="Max validation samples (for quick testing)")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("COCO Captioning Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Device: {device}")
    print(f"Num steps: {args.num_steps}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load data config
    config_path = os.path.join(args.data_dir, "config.json")
    with open(config_path) as f:
        data_config = json.load(f)

    # Create model
    decoder_config_dict = checkpoint['decoder_config']
    decoder_config = ModelConfig(**decoder_config_dict)

    model = DiffusionTransformer(decoder_config)
    model.load_state_dict(checkpoint['decoder_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model: {decoder_config.n_layers} layers, d_model={decoder_config.d_model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create diffusion
    diffusion = DiscreteDiffusion(
        vocab_size=decoder_config.vocab_size,
        mask_token_id=3,
        pad_token_id=0,
        schedule="cosine",
    )

    # Load tokenizer
    tokenizer_path = data_config.get('tokenizer', 'data_full/tokenizer.json')
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Tokenizer: {tokenizer_path}")

    # Load validation data
    print("\nLoading validation data...")
    val_features = torch.load(os.path.join(args.data_dir, "val_image_features.pt"))
    val_captions = torch.load(os.path.join(args.data_dir, "val_captions.pt"))
    val_captions_json = json.load(open(os.path.join(args.data_dir, "val_captions.json")))

    # Limit samples if requested
    if args.max_samples:
        val_features = val_features[:args.max_samples]
        val_captions = val_captions[:args.max_samples]
        val_captions_json = val_captions_json[:args.max_samples]

    print(f"Validation samples: {len(val_captions)}")

    # Create dataset
    val_dataset = CaptionDataset(val_features, val_captions)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Generate captions
    print("\nGenerating captions...")
    generated_captions = {}
    reference_captions = {}

    idx = 0
    for batch in tqdm(val_loader, desc="Generating"):
        image_features = batch['image_features'].to(device)

        # Generate
        captions = generate_captions(
            model,
            diffusion,
            image_features,
            tokenizer,
            num_steps=args.num_steps,
            temperature=args.temperature,
            device=device,
        )

        # Store results
        for cap in captions:
            generated_captions[idx] = cap

            # Get reference captions
            if idx < len(val_captions_json):
                # val_captions_json is a list of strings (one caption per image)
                ref_caption = val_captions_json[idx]
                if isinstance(ref_caption, str):
                    reference_captions[idx] = [ref_caption]  # Wrap in list for COCO eval format
                else:
                    # If it's already a list or dict, handle accordingly
                    reference_captions[idx] = [ref_caption] if isinstance(ref_caption, str) else ref_caption

            idx += 1

    print(f"\nGenerated {len(generated_captions)} captions")

    # Save generated captions
    output_path = os.path.join(args.output_dir, "generated_captions.json")
    with open(output_path, 'w') as f:
        json.dump(generated_captions, f, indent=2)
    print(f"Saved generated captions to: {output_path}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = evaluate_coco_metrics(generated_captions, reference_captions)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if metrics:
        for metric_name, score in metrics.items():
            print(f"{metric_name:12s}: {score:6.2f}")
    else:
        print("No metrics computed (pycocoevalcap not installed)")

    print("=" * 70)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    # Print some example captions
    print("\n" + "=" * 70)
    print("EXAMPLE CAPTIONS (first 5)")
    print("=" * 70)

    for i in range(min(5, len(generated_captions))):
        print(f"\nImage {i}:")
        print(f"  Generated: {generated_captions[i]}")
        if i in reference_captions:
            print(f"  References:")
            for ref in reference_captions[i][:3]:  # Show max 3 references
                print(f"    - {ref}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
