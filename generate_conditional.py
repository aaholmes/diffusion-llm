#!/usr/bin/env python3
"""
Generate text from a conditional diffusion model.

Takes a first sentence as input and generates the continuation.

Usage:
    python generate_conditional.py --checkpoint checkpoints_conditional/best.pt
    python generate_conditional.py --prompt "Once upon a time there was a little cat."
"""

import argparse

import torch
from tokenizers import Tokenizer

from model import DiffusionTransformer, TextEncoder, ModelConfig


def load_conditional_model(checkpoint_path: str, device: str = "cpu"):
    """Load encoder and decoder from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load encoder
    encoder_config = checkpoint['encoder_config']
    if isinstance(encoder_config, dict):
        encoder_config = ModelConfig(**encoder_config)
    encoder = TextEncoder(encoder_config)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()

    # Load decoder
    decoder_config = checkpoint['decoder_config']
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)
    decoder = DiffusionTransformer(decoder_config)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.to(device)
    decoder.eval()

    print(f"  Encoder: {encoder_config.n_layers} layers, d_model={encoder_config.d_model}")
    print(f"  Decoder: {decoder_config.n_layers} layers, d_model={decoder_config.d_model}")

    return encoder, decoder, checkpoint['train_config']


@torch.no_grad()
def generate(
    encoder,
    decoder,
    prompt_tokens: torch.Tensor,
    max_len: int = 192,
    steps: int = 100,
    temperature: float = 0.8,
    device: str = "cpu",
    mask_token_id: int = 3,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Generate continuation given prompt tokens."""
    prompt_tokens = prompt_tokens.to(device)
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    batch_size = prompt_tokens.shape[0]
    vocab_size = decoder.vocab_size

    # Encode prompt
    enc_attention_mask = (prompt_tokens != pad_token_id).float()
    encoder_output = encoder(prompt_tokens, enc_attention_mask)

    # Start with all masks
    x = torch.full((batch_size, max_len), mask_token_id, device=device)

    # Iteratively denoise
    for step in range(steps):
        t = torch.tensor([1.0 - step / steps] * batch_size, device=device)

        logits = decoder(
            x, t,
            encoder_output=encoder_output,
            encoder_attention_mask=enc_attention_mask,
        )

        # Sample from logits
        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1)
        sampled = sampled.view(batch_size, -1)

        # Unmask based on timestep
        mask_rate = 1.0 - (step + 1) / steps
        num_to_keep_masked = int(mask_rate * max_len)

        is_masked = (x == mask_token_id)

        for b in range(batch_size):
            masked_indices = is_masked[b].nonzero().squeeze(-1)
            if len(masked_indices.shape) == 0:
                masked_indices = masked_indices.unsqueeze(0)
            if len(masked_indices) > num_to_keep_masked:
                perm = torch.randperm(len(masked_indices), device=device)
                to_unmask = masked_indices[perm[num_to_keep_masked:]]
                x[b, to_unmask] = sampled[b, to_unmask]

    return x


def main():
    parser = argparse.ArgumentParser(description="Generate from conditional model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_conditional/best.pt")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a little girl.")
    parser.add_argument("--num_samples", "-n", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--max_len", type=int, default=192)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    mask_id = tokenizer.token_to_id("<MASK>")

    # Load model
    encoder, decoder, train_config = load_conditional_model(args.checkpoint, device)
    max_encoder_len = train_config.get('max_encoder_len', 64)

    # Tokenize prompt
    print(f"\nPrompt: {args.prompt}")
    encoded = tokenizer.encode(args.prompt)
    prompt_ids = [bos_id] + encoded.ids + [eos_id]

    # Pad to max encoder length
    if len(prompt_ids) > max_encoder_len:
        prompt_ids = prompt_ids[:max_encoder_len]
    else:
        prompt_ids = prompt_ids + [pad_id] * (max_encoder_len - len(prompt_ids))

    prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).repeat(args.num_samples, 1)

    # Generate
    print(f"\nGenerating {args.num_samples} samples with {args.steps} steps, temp={args.temperature}...")
    output = generate(
        encoder, decoder, prompt_tensor,
        max_len=args.max_len,
        steps=args.steps,
        temperature=args.temperature,
        device=device,
        mask_token_id=mask_id,
        pad_token_id=pad_id,
    )

    # Decode and print
    print("\n" + "=" * 60)
    print("Generated Continuations:")
    print("=" * 60)

    for i in range(args.num_samples):
        tokens = output[i].tolist()
        # Remove padding and special tokens
        tokens = [t for t in tokens if t not in [pad_id, bos_id, eos_id, mask_id]]
        text = tokenizer.decode(tokens)
        print(f"\n[{i+1}] {text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
