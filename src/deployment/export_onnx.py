#!/usr/bin/env python3
"""
Export the captioning model to ONNX format for Jetson deployment.

Usage:
    # Export decoder only (CLIP runs separately)
    python export_onnx.py --checkpoint checkpoints_caption_poc/final.pt --output model_decoder.onnx

    # Test the exported model
    python export_onnx.py --checkpoint checkpoints_caption_poc/final.pt --test
"""

import argparse
import os

import torch
import torch.nn as nn

from src.core.model import DiffusionTransformer, ModelConfig


class DecoderWrapper(nn.Module):
    """
    Wrapper for the decoder that handles ONNX export requirements.

    ONNX doesn't support dynamic control flow well, so we export
    a single forward pass that can be called iteratively.
    """

    def __init__(self, decoder: DiffusionTransformer):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,           # [batch, seq_len] - token ids
        t: torch.Tensor,           # [batch] - timestep
        encoder_output: torch.Tensor,  # [batch, enc_seq, d_model] - CLIP features
    ) -> torch.Tensor:
        """
        Single denoising step.

        Args:
            x: Current token sequence (may contain mask tokens)
            t: Current timestep (1.0 = all noise, 0.0 = clean)
            encoder_output: Pre-computed CLIP image features

        Returns:
            logits: [batch, seq_len, vocab_size] - token predictions
        """
        # Create attention mask (all ones for CLIP features)
        encoder_mask = torch.ones(
            encoder_output.shape[:2],
            device=encoder_output.device,
            dtype=encoder_output.dtype
        )

        logits = self.decoder(
            x, t,
            encoder_output=encoder_output,
            encoder_attention_mask=encoder_mask,
        )

        return logits


def load_decoder(checkpoint_path: str, device: str = "cpu"):
    """Load decoder from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    decoder_config = checkpoint['decoder_config']
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)

    decoder = DiffusionTransformer(decoder_config)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    return decoder, decoder_config


def export_to_onnx(
    decoder: DiffusionTransformer,
    config: ModelConfig,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 64,
    encoder_seq_len: int = 50,
    opset_version: int = 14,
):
    """Export decoder to ONNX format."""
    print(f"\nExporting to ONNX: {output_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Encoder sequence length: {encoder_seq_len}")
    print(f"  ONNX opset version: {opset_version}")

    # Wrap decoder
    wrapper = DecoderWrapper(decoder)
    wrapper.eval()

    # Create dummy inputs
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    t = torch.rand(batch_size)
    encoder_output = torch.randn(batch_size, encoder_seq_len, config.d_model)

    # Export
    torch.onnx.export(
        wrapper,
        (x, t, encoder_output),
        output_path,
        input_names=['tokens', 'timestep', 'image_features'],
        output_names=['logits'],
        dynamic_axes={
            'tokens': {0: 'batch'},
            'timestep': {0: 'batch'},
            'image_features': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"  Exported successfully!")

    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.1f} MB")

    return output_path


def verify_onnx(onnx_path: str, config: ModelConfig, batch_size: int = 1, seq_len: int = 64, encoder_seq_len: int = 50):
    """Verify the ONNX model works correctly."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime not installed. Run: pip install onnxruntime")
        return False

    print(f"\nVerifying ONNX model: {onnx_path}")

    # Create session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Get input/output info
    print("  Inputs:")
    for inp in session.get_inputs():
        print(f"    {inp.name}: {inp.shape} ({inp.type})")

    print("  Outputs:")
    for out in session.get_outputs():
        print(f"    {out.name}: {out.shape} ({out.type})")

    # Run inference
    import numpy as np

    x = np.random.randint(0, config.vocab_size, (batch_size, seq_len)).astype(np.int64)
    t = np.random.rand(batch_size).astype(np.float32)
    encoder_output = np.random.randn(batch_size, encoder_seq_len, config.d_model).astype(np.float32)

    outputs = session.run(
        None,
        {'tokens': x, 'timestep': t, 'image_features': encoder_output}
    )

    logits = outputs[0]
    print(f"\n  Test inference successful!")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    return True


def benchmark_onnx(onnx_path: str, config: ModelConfig, num_runs: int = 100, seq_len: int = 64, encoder_seq_len: int = 50):
    """Benchmark ONNX inference speed."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime not installed")
        return

    import time
    import numpy as np

    print(f"\nBenchmarking ONNX model ({num_runs} runs)...")

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Prepare inputs
    batch_size = 1

    x = np.random.randint(0, config.vocab_size, (batch_size, seq_len)).astype(np.int64)
    t = np.random.rand(batch_size).astype(np.float32)
    encoder_output = np.random.randn(batch_size, encoder_seq_len, config.d_model).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {'tokens': x, 'timestep': t, 'image_features': encoder_output})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {'tokens': x, 'timestep': t, 'image_features': encoder_output})
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000

    print(f"  Average latency: {avg_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  Throughput: {1000/avg_ms:.1f} steps/sec")
    print(f"  For 50 diffusion steps: {avg_ms * 50 / 1000:.2f} sec total")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="model_decoder.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Max sequence length for captions")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--test", action="store_true",
                        help="Test the exported model")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference speed")
    args = parser.parse_args()

    # Load model
    decoder, config = load_decoder(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_config = checkpoint.get('train_config', {})

    print(f"\nModel config:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  vocab_size: {config.vocab_size}")

    # Export
    export_to_onnx(
        decoder, config, args.output,
        seq_len=args.seq_len,
        opset_version=args.opset,
    )

    # Save config JSON for inference
    config_path = args.output.replace('.onnx', '_config.json')
    config_dict = {
        'd_model': config.d_model,
        'n_layers': config.n_layers,
        'n_heads': config.n_heads,
        'vocab_size': config.vocab_size,
        'max_seq_len': config.max_seq_len,
        'dropout': config.dropout,
        'max_caption_len': train_config.get('max_caption_len', args.seq_len),
    }

    import json
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"\n  Saved config to: {config_path}")

    # Verify
    if args.test or args.benchmark:
        if verify_onnx(args.output, config, seq_len=args.seq_len):
            if args.benchmark:
                benchmark_onnx(args.output, config, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
