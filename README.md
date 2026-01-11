# diffusion-llm

A discrete diffusion language model built from scratch with PyTorch, trained on TinyStories and optimized for Jetson Orin Nano deployment.

## Overview

Unlike autoregressive models (GPT-style) that generate text left-to-right, diffusion LMs start with masked tokens and iteratively "unmask" them using bidirectional context. This enables parallel generation and native infilling capabilities.

## Project Status

- [x] **Phase 1**: Data preparation (tokenizer, dataset processing)
- [x] **Phase 2**: Model architecture (bidirectional transformer, diffusion process)
- [ ] **Phase 3**: Training loop
- [ ] **Phase 4**: Generation & evaluation
- [ ] **Phase 5**: Jetson optimization
- [ ] **Phase 6**: Extensions (multimodal, custom CUDA kernels)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (downloads TinyStories, trains tokenizer, tokenizes dataset)
python data_prep.py

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

## Model Configurations

| Config | Parameters | Size | Use Case |
|--------|------------|------|----------|
| tiny   | 7.6M       | 29 MB | Debugging |
| small  | 17.3M      | 66 MB | Prototyping |
| medium | 34.3M      | 131 MB | Production |
| large  | 60.7M      | 232 MB | Best quality |

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

**Current coverage: 98% (119 tests passing)**

| Module | Coverage |
|--------|----------|
| `data_prep.py` | 96% |
| `model.py` | 99% |
| `diffusion.py` | 100% |
