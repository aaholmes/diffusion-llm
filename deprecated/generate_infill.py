#!/usr/bin/env python3
"""
Generate infilled text from first and last sentences using trained conditional model.

Usage:
    python generate_infill.py --checkpoint checkpoints_infill_extended/final.pt
    python generate_infill.py --first "Once upon a time there was a little cat." --last "They all lived happily ever after."
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

    return encoder, decoder, checkpoint.get('train_config', {})


@torch.no_grad()
def generate_infill(
    encoder,
    decoder,
    condition_tokens: torch.Tensor,
    max_len: int = 160,
    steps: int = 100,
    temperature: float = 0.8,
    device: str = "cpu",
    mask_token_id: int = 3,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Generate middle text given first + last sentences (condition)."""
    condition_tokens = condition_tokens.to(device)
    if condition_tokens.dim() == 1:
        condition_tokens = condition_tokens.unsqueeze(0)

    batch_size = condition_tokens.shape[0]
    vocab_size = decoder.vocab_size

    # Encode condition (first + last sentences)
    enc_attention_mask = (condition_tokens != pad_token_id).float()
    encoder_output = encoder(condition_tokens, enc_attention_mask)

    # Start with all masks
    x = torch.full((batch_size, max_len), mask_token_id, device=device)

    # Iteratively denoise
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for step in range(steps):
        t = timesteps[step].expand(batch_size)
        t_next = timesteps[step + 1].expand(batch_size)

        logits = decoder(
            x, t,
            encoder_output=encoder_output,
            encoder_attention_mask=enc_attention_mask,
        )

        # Sample from logits
        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1)
        sampled = sampled.view(batch_size, max_len)

        # Unmask based on timestep (cosine schedule)
        current_mask_rate = 1 - torch.cos(t * torch.pi / 2)
        next_mask_rate = 1 - torch.cos(t_next * torch.pi / 2)

        keep_mask_prob = (next_mask_rate / current_mask_rate.clamp(min=1e-8)).clamp(max=1.0)

        is_masked = (x == mask_token_id)
        rand = torch.rand(batch_size, max_len, device=device)
        keep_mask = rand < keep_mask_prob.unsqueeze(1)

        unmask = is_masked & ~keep_mask
        x[unmask] = sampled[unmask]

    return x


def main():
    parser = argparse.ArgumentParser(description="Generate infilled text from first+last sentences")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_infill_extended/final.pt")
    parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
    parser.add_argument("--first", type=str, default="Once upon a time there was a little cat.")
    parser.add_argument("--last", type=str, default="The cat was very happy and went home.")
    parser.add_argument("--num_samples", "-n", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", "-t", type=float, default=0.9)
    parser.add_argument("--max_len", type=int, default=160)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    sep_id = eos_id  # SEP is same as EOS in our tokenizer
    mask_id = tokenizer.token_to_id("<MASK>")

    # Load model
    encoder, decoder, train_config = load_conditional_model(args.checkpoint, device)
    max_encoder_len = train_config.get('max_encoder_len', 96)

    # Tokenize first and last sentences
    print(f"First sentence: {args.first}")
    print(f"Last sentence:  {args.last}\n")

    first_enc = tokenizer.encode(args.first)
    last_enc = tokenizer.encode(args.last)

    # Build condition: BOS + first + SEP + last + EOS
    condition_ids = [bos_id] + first_enc.ids + [sep_id] + last_enc.ids + [eos_id]

    # Pad to max encoder length
    if len(condition_ids) > max_encoder_len:
        condition_ids = condition_ids[:max_encoder_len]
    else:
        condition_ids = condition_ids + [pad_id] * (max_encoder_len - len(condition_ids))

    condition_tensor = torch.tensor(condition_ids).unsqueeze(0).repeat(args.num_samples, 1)

    # Generate
    print(f"Generating {args.num_samples} samples with {args.steps} steps, temp={args.temperature}...")
    output = generate_infill(
        encoder, decoder, condition_tensor,
        max_len=args.max_len,
        steps=args.steps,
        temperature=args.temperature,
        device=device,
        mask_token_id=mask_id,
        pad_token_id=pad_id,
    )

    # Decode and print
    print("\n" + "=" * 70)
    print("Generated Middle Text:")
    print("=" * 70)

    for i in range(args.num_samples):
        tokens = output[i].tolist()
        # Remove padding and special tokens
        tokens = [t for t in tokens if t not in [pad_id, bos_id, eos_id, mask_id, sep_id]]
        text = tokenizer.decode(tokens)

        print(f"\n[{i+1}] {args.first} {text} {args.last}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
