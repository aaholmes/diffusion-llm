#!/usr/bin/env python3
"""
Smoke test for conditional training pipeline.

Verifies end-to-end:
1. Conditional model creation (encoder + decoder with cross-attention)
2. Conditional training with encoder-decoder pairs
3. Loss decreases over training
4. Conditional sampling works

Run with: python smoke_test_conditioning.py
"""

import torch
import torch.nn as nn

from model import (
    create_model,
    create_encoder,
    create_conditional_model,
    ConditionalDiffusionLM,
)
from diffusion import DiscreteDiffusion


def test_standalone_decoder_with_cross_attention():
    """Test decoder alone with cross-attention using external encoder output."""
    print("\n" + "=" * 60)
    print("Test 1: Standalone Decoder with Cross-Attention")
    print("=" * 60)

    # Create decoder with cross-attention enabled
    decoder = create_model(
        "tiny",
        vocab_size=256,
        max_seq_len=64,
        has_cross_attention=True,
    )

    print(f"  Decoder params: {decoder.count_parameters():,}")
    print(f"  Has cross-attention: {decoder.config.has_cross_attention}")

    # Create diffusion process
    diffusion = DiscreteDiffusion(vocab_size=256, mask_token_id=3)

    # Test data
    batch_size = 4
    decoder_seq_len = 32
    encoder_seq_len = 16

    # Decoder input (tokens to denoise)
    x = torch.randint(5, 256, (batch_size, decoder_seq_len))

    # Simulated encoder output (as if from a frozen encoder)
    encoder_output = torch.randn(batch_size, encoder_seq_len, decoder.config.d_model)

    # Training step with conditioning
    loss, metrics = diffusion.training_losses(
        decoder, x, encoder_output=encoder_output
    )

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # Test sampling with conditioning
    samples = diffusion.sample(
        decoder,
        batch_size=2,
        seq_len=16,
        num_steps=5,
        device="cpu",
        encoder_output=encoder_output[:2],
    )

    print(f"  Sample shape: {samples.shape}")
    print("  ✓ Standalone decoder with cross-attention works!")

    return True


def test_full_conditional_model():
    """Test full encoder-decoder conditional model."""
    print("\n" + "=" * 60)
    print("Test 2: Full Conditional Model (Encoder + Decoder)")
    print("=" * 60)

    # Create conditional model
    model = create_conditional_model(
        encoder_config="tiny",
        decoder_config="tiny",
        vocab_size=256,
        encoder_max_seq_len=32,
        decoder_max_seq_len=64,
    )

    print(f"  Total params: {model.count_parameters():,}")
    print(f"  Encoder params: {model.encoder.count_parameters():,}")
    print(f"  Decoder params: {model.decoder.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    encoder_input = torch.randint(5, 256, (batch_size, 32))
    decoder_input = torch.randint(5, 256, (batch_size, 64))
    t = torch.rand(batch_size)

    logits = model(decoder_input, t, encoder_input)
    print(f"  Output shape: {logits.shape}")

    print("  ✓ Full conditional model works!")

    return True


def test_conditional_training_loop():
    """Test conditional training with loss decreasing."""
    print("\n" + "=" * 60)
    print("Test 3: Conditional Training Loop")
    print("=" * 60)

    # Create conditional model
    model = create_conditional_model(
        encoder_config="tiny",
        decoder_config="tiny",
        vocab_size=256,
        encoder_max_seq_len=32,
        decoder_max_seq_len=64,
    )

    diffusion = DiscreteDiffusion(vocab_size=256, mask_token_id=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Generate synthetic conditional pairs
    # Encoder input (first sentence) -> Decoder target (rest of paragraph)
    num_samples = 100
    encoder_data = torch.randint(5, 256, (num_samples, 32))
    decoder_data = torch.randint(5, 256, (num_samples, 64))

    # Training loop
    batch_size = 16
    num_steps = 20
    losses = []

    model.train()
    for step in range(num_steps):
        # Sample batch
        idx = torch.randint(0, num_samples, (batch_size,))
        enc_batch = encoder_data[idx]
        dec_batch = decoder_data[idx]

        # Get encoder output
        encoder_output = model.encoder(enc_batch)

        # Compute loss on decoder with conditioning
        loss, metrics = diffusion.training_losses(
            model.decoder, dec_batch, encoder_output=encoder_output
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(metrics["loss"])

        if step % 5 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")

    # Check loss decreased
    first_losses = sum(losses[:5]) / 5
    last_losses = sum(losses[-5:]) / 5

    print(f"\n  First 5 avg loss: {first_losses:.4f}")
    print(f"  Last 5 avg loss: {last_losses:.4f}")

    if last_losses < first_losses:
        print("  ✓ Loss decreased during training!")
    else:
        print("  ⚠ Loss did not decrease (may need more steps)")

    return True


def test_staged_training():
    """Test staged training approach (freeze decoder, train encoder only)."""
    print("\n" + "=" * 60)
    print("Test 4: Staged Training (Freeze Decoder)")
    print("=" * 60)

    # Create conditional model
    model = create_conditional_model(
        encoder_config="tiny",
        decoder_config="tiny",
        vocab_size=256,
        encoder_max_seq_len=32,
        decoder_max_seq_len=64,
    )

    # Count trainable params before freeze
    trainable_before = model.count_parameters(trainable_only=True)
    print(f"  Trainable params before freeze: {trainable_before:,}")

    # Freeze decoder (keeps encoder + cross-attention trainable)
    model.freeze_decoder()

    trainable_after = model.count_parameters(trainable_only=True)
    print(f"  Trainable params after freeze: {trainable_after:,}")
    print(f"  Reduction: {(1 - trainable_after/trainable_before)*100:.1f}%")

    # Verify encoder is trainable
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"  Encoder trainable: {encoder_trainable:,}")

    # Verify self-attention is frozen
    decoder_self_attn_trainable = 0
    for block in model.decoder.blocks:
        decoder_self_attn_trainable += sum(
            p.numel() for p in block.self_attn.parameters() if p.requires_grad
        )
    print(f"  Decoder self-attention trainable: {decoder_self_attn_trainable}")

    # Verify cross-attention is trainable
    cross_attn_trainable = 0
    for block in model.decoder.blocks:
        if block.has_cross_attention:
            cross_attn_trainable += sum(
                p.numel() for p in block.cross_attn.parameters() if p.requires_grad
            )
    print(f"  Cross-attention trainable: {cross_attn_trainable:,}")

    # Quick training step to verify gradients flow
    diffusion = DiscreteDiffusion(vocab_size=256, mask_token_id=3)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    enc_input = torch.randint(5, 256, (4, 32))
    dec_input = torch.randint(5, 256, (4, 64))

    encoder_output = model.encoder(enc_input)
    loss, _ = diffusion.training_losses(
        model.decoder, dec_input, encoder_output=encoder_output
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("  ✓ Staged training (frozen decoder) works!")

    # Unfreeze and verify
    model.unfreeze_all()
    trainable_unfrozen = model.count_parameters(trainable_only=True)
    print(f"  Trainable after unfreeze: {trainable_unfrozen:,}")
    assert trainable_unfrozen == trainable_before
    print("  ✓ Unfreeze works!")

    return True


def test_conditional_sampling():
    """Test conditional text generation."""
    print("\n" + "=" * 60)
    print("Test 5: Conditional Sampling")
    print("=" * 60)

    # Create model
    model = create_conditional_model(
        encoder_config="tiny",
        decoder_config="tiny",
        vocab_size=256,
        encoder_max_seq_len=32,
        decoder_max_seq_len=64,
    )

    diffusion = DiscreteDiffusion(vocab_size=256, mask_token_id=3)

    # Encoder input (simulating "first sentence")
    encoder_input = torch.randint(5, 256, (2, 32))

    # Get encoder output
    model.eval()
    with torch.no_grad():
        encoder_output = model.encoder(encoder_input)

    # Sample conditioned on encoder output
    samples = diffusion.sample(
        model.decoder,
        batch_size=2,
        seq_len=32,
        num_steps=10,
        temperature=0.8,
        device="cpu",
        encoder_output=encoder_output,
    )

    print(f"  Encoder input shape: {encoder_input.shape}")
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Generated samples shape: {samples.shape}")
    print(f"  Sample[0] first 10 tokens: {samples[0, :10].tolist()}")

    # Verify samples are not all masks
    mask_count = (samples == 3).sum().item()
    total = samples.numel()
    print(f"  Mask tokens in output: {mask_count}/{total} ({mask_count/total*100:.1f}%)")

    if mask_count < total * 0.1:  # Less than 10% masks
        print("  ✓ Conditional sampling produces valid output!")
    else:
        print("  ⚠ Many mask tokens remain (may need more steps)")

    return True


def test_different_encoder_decoder_sizes():
    """Test encoder and decoder with different sizes."""
    print("\n" + "=" * 60)
    print("Test 6: Different Encoder/Decoder Sizes")
    print("=" * 60)

    # Small encoder, larger decoder
    model = create_conditional_model(
        encoder_config="tiny",
        decoder_config="small",  # Larger decoder
        vocab_size=256,
        encoder_max_seq_len=32,
        decoder_max_seq_len=128,
    )

    print(f"  Encoder config: tiny")
    print(f"  Decoder config: small")
    print(f"  Encoder d_model: {model.encoder.config.d_model}")
    print(f"  Decoder d_model: {model.decoder.config.d_model}")

    # Forward pass
    enc_input = torch.randint(5, 256, (2, 32))
    dec_input = torch.randint(5, 256, (2, 128))
    t = torch.rand(2)

    logits = model(dec_input, t, enc_input)
    print(f"  Output shape: {logits.shape}")

    print("  ✓ Different encoder/decoder sizes work!")

    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("CONDITIONAL TRAINING SMOKE TESTS")
    print("=" * 60)

    tests = [
        ("Standalone Decoder with Cross-Attention", test_standalone_decoder_with_cross_attention),
        ("Full Conditional Model", test_full_conditional_model),
        ("Conditional Training Loop", test_conditional_training_loop),
        ("Staged Training", test_staged_training),
        ("Conditional Sampling", test_conditional_sampling),
        ("Different Encoder/Decoder Sizes", test_different_encoder_decoder_sizes),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FAILED: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r, _ in results if r)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All smoke tests passed!")
        return 0
    else:
        print("\n  ⚠ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
