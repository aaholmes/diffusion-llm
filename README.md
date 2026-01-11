# diffusion-llm

A discrete diffusion language model built from scratch with PyTorch, trained on TinyStories and optimized for Jetson Orin Nano deployment.

## Overview

Unlike autoregressive models (GPT-style) that generate text left-to-right, diffusion LMs start with masked tokens and iteratively "unmask" them using bidirectional context. This enables parallel generation and native infilling capabilities.

## Project Status

- [x] **Phase 1**: Data preparation (tokenizer, dataset processing)
- [x] **Phase 2**: Model architecture (bidirectional transformer, diffusion process)
- [x] **Phase 3**: Training loop (mixed precision, checkpointing, wandb)
- [x] **Phase 4**: Conditioning architecture (encoder, cross-attention, staged training)
- [ ] **Phase 5**: Jetson optimization
- [ ] **Phase 6**: Extensions (multimodal, custom CUDA kernels)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (downloads TinyStories, trains tokenizer, tokenizes dataset)
python data_prep.py

# Train model
python train.py --model_config small --max_steps 50000

# Run tests
pytest -v
```

## Architecture

**DiffusionTransformer** (`model.py`):
- Bidirectional transformer with no causal masking
- Sinusoidal timestep embeddings
- Pre-norm transformer blocks
- Configurable sizes (tiny → large)

**DiscreteDiffusion** (`diffusion.py`):
- Cosine noise schedule for smooth masking
- Forward process: progressively mask tokens
- Reverse process: iteratively unmask via model predictions

**Trainer** (`train.py`):
- AdamW optimizer with linear warmup + cosine decay
- Mixed precision training (FP16)
- Gradient accumulation and clipping
- Automatic checkpointing with best model tracking
- Optional Weights & Biases logging

## Model Configurations

| Config  | d_model | Heads | Layers | Parameters | Use Case |
|---------|---------|-------|--------|------------|----------|
| tiny    | 256     | 4     | 4      | ~4.5M      | Debugging |
| small   | 384     | 6     | 6      | ~15M       | Prototyping |
| medium  | 512     | 8     | 8      | ~35M       | Production |
| large   | 640     | 10    | 10     | ~60M       | Best quality |
| xlarge  | 768     | 12    | 12     | ~110M      | Jetson (~1.2s) |
| xxlarge | 1024    | 16    | 16     | ~250M      | Jetson (~2.0s) |

## Hardware Targets

- **Training**: Desktop GPU (RTX 5060 Ti 16GB)
- **Inference**: Jetson Orin Nano 8GB

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=term-missing
```

**Current coverage: 94% (202 tests passing)**

| Module | Coverage |
|--------|----------|
| `data_prep.py` | 96% |
| `model.py` | 97% |
| `diffusion.py` | 100% |
| `train.py` | 87% |

## Conditioning Architecture

Phase 4 adds encoder-decoder conditioning for controlled generation:

- **TextEncoder**: Bidirectional transformer encoding input text
- **Cross-Attention**: Decoder attends to encoder output (original Transformer style)
- **Staged Training**: Train denoiser → freeze → train encoder + cross-attention
- **Data Pipeline**: Extract (first sentence → rest of paragraph) pairs

```python
from model import create_conditional_model

# Create encoder-decoder model
model = create_conditional_model(
    encoder_config="small",
    decoder_config="small",
    vocab_size=8192,
)

# Stage 2: Freeze decoder, train only encoder + cross-attention
model.freeze_decoder()
```

## Future: Block Diffusion

Long-term stretch goal: implement [Block Diffusion](https://arxiv.org/abs/2503.09573) (ICLR 2025 Oral) which interpolates between autoregressive and diffusion models:

- **Block-wise generation**: Divide sequence into blocks, generate blocks autoregressively, diffuse within each block
- **Arbitrary length**: No fixed sequence length limitation
- **KV caching**: Cache across blocks for faster inference
- **Tunable tradeoff**: Block size controls quality vs speed

This would combine parallel generation (fast) with streaming output (responsive UX) — ideal for Jetson deployment.
