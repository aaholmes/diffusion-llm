# diffusion-llm

A discrete diffusion language model built from scratch with PyTorch, trained on TinyStories and optimized for Jetson Orin Nano deployment.

## Overview

Unlike autoregressive models (GPT-style) that generate text left-to-right, diffusion LMs start with masked tokens and iteratively "unmask" them using bidirectional context. This enables parallel generation and native infilling capabilities.

## Project Status

- [x] **Phase 1**: Data preparation (tokenizer, dataset processing)
- [ ] **Phase 2**: Model architecture (bidirectional transformer)
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

# With custom settings
python data_prep.py --subset_size 500000 --vocab_size 16384
```

## Hardware Targets

- **Training**: Desktop GPU (RTX 5060 Ti 16GB)
- **Inference**: Jetson Orin Nano 8GB

## Model Configurations

| Config | Parameters | Use Case |
|--------|------------|----------|
| tiny   | ~10M       | Debugging |
| small  | ~25M       | Prototyping |
| medium | ~50M       | Production |
| large  | ~80M       | Best quality |
