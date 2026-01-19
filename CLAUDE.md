# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **discrete diffusion language models** from scratch, exploring different architectures and training approaches. The project demonstrates:

- Bidirectional transformer architecture for discrete diffusion
- MDLM-style masking/unmasking diffusion process
- Sparse Distribution Diffusion (SDD) - a novel approach using sparse probability distributions
- Full pipeline: tokenization → training → inference

Target model sizes: 25-80M parameters, trained on TinyStories dataset.

## Key Commands

### Environment Setup

```bash
conda create -n diffusion-lm python=3.11
conda activate diffusion-lm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Data Preparation

```bash
python src/data/data_prep.py    # Download TinyStories, train tokenizer, tokenize dataset
```

**Outputs:**
- `tokenizer.json` - BPE tokenizer with 8192 vocab
- `tokenized_data.pt` - Tensor of tokenized sequences

### Training

```bash
python src/training/train.py              # Train standard diffusion model
python src/training/train_bilateral.py    # Train SDD model with bilateral attention
```

**Key parameters**:
- `model_config`: "tiny" (10M), "small" (25M), "medium" (50M), "large" (80M)
- `batch_size`: 64 (adjust based on GPU memory)
- `max_steps`: 20000 (prototype), 50000 (full training)
- `learning_rate`: 3e-4
- `use_wandb`: True (for experiment tracking)

### Generation

```bash
python src/generation/generate.py    # Interactive text generation
```

### Testing

```bash
pytest tests/                        # Run all tests
pytest --cov=src --cov-fail-under=70 # Run with coverage
```

## Architecture

### Core Components

The codebase follows a modular structure in `src/`:

**1. Data Pipeline** (`src/data/data_prep.py`)
- Downloads TinyStories dataset
- Trains BPE tokenizer (vocab_size=8192)
- Tokenizes and pads sequences to fixed length (default 256)
- Special tokens: `<PAD>` (0), `<BOS>` (1), `<EOS>` (2), `<MASK>` (3), `<UNK>` (4)

**2. Model Architecture** (`src/core/model.py`)
- **DiffusionTransformer**: Bidirectional transformer with timestep conditioning
  - Token embedding + positional embedding
  - Sinusoidal timestep embeddings (added to all positions)
  - N transformer blocks with **bidirectional attention** (no causal masking)
  - Output projection to vocabulary logits
- **Key difference from autoregressive LLMs**: Uses bidirectional attention, conditions on noise level `t`, predicts entire sequence at once

**3. Diffusion Process** (`src/core/diffusion.py`)
- **DiscreteDiffusion**: Implements discrete masking diffusion
  - **Forward process** (`q_sample`): Gradually mask tokens based on noise schedule
  - **Reverse process** (`p_sample`): Iteratively unmask tokens using model predictions
  - **Noise schedule**: Cosine or linear (cosine is smoother near endpoints)
  - **Training**: Cross-entropy loss only on masked positions
  - **Sampling**: Start from all masks, denoise over N steps (typically 25-50)

**4. Sparse Distribution Diffusion** (`src/core/sparse_diffusion.py`, `src/core/sparse_model.py`)
- **SparseState**: Represents probability distributions with only top-k candidates per position
- **SparseDiffusion**: Noise injection and denoising with sparse distributions
- **BilateralSparseDenoiser**: Transformer with intra-position and inter-position attention
  - Intra-position: candidates at same position attend to each other
  - Inter-position: candidates attend across positions

**5. Training** (`src/training/train.py`, `src/training/train_bilateral.py`)
- Mixed precision training (FP16) with gradient accumulation
- AdamW optimizer with linear warmup + cosine decay
- Model compilation via `torch.compile()` for speed (PyTorch 2.0+)
- Wandb integration for experiment tracking
- Checkpointing every N steps

**6. Generation** (`src/generation/generate.py`)
- Interactive text generation with temperature and top-k sampling
- Supports conditional generation (prompt-based)
- Multiple samples per prompt

### Model Configurations

Defined in `src/core/model.py`:

| Config | Params | d_model | n_heads | n_layers | d_ff |
|--------|--------|---------|---------|----------|------|
| tiny   | ~7.6M  | 256     | 4       | 4        | 1024 |
| small  | ~17M   | 384     | 6       | 6        | 1536 |
| medium | ~34M   | 512     | 8       | 8        | 2048 |
| large  | ~61M   | 640     | 10      | 10       | 2560 |
| xlarge | ~110M  | 768     | 12      | 12       | 3072 |

### Diffusion Process Details

**Forward Process (Noise Injection):**
1. Given clean tokens `x` and noise level `t ∈ [0,1]`
2. Compute mask rate: `rate = get_mask_rate(t)` (0 at t=0, 1 at t=1)
3. Randomly mask `rate` fraction of tokens with `<MASK>` token
4. Padding tokens are never masked

**Reverse Process (Denoising):**
1. Start with all `<MASK>` tokens (or prompt + masks)
2. For t from 1.0 to 0.0 in N steps:
   - Run model to predict logits for all positions
   - Sample tokens from predicted distribution
   - Determine which masks to keep based on t → t_next schedule
   - Unmask some positions, keep others masked
3. Final output: fully unmasked sequence

**Key Insight**: Unlike autoregressive models that generate left-to-right, diffusion models iteratively refine all positions in parallel. This enables:
- Faster generation for long sequences (parallelism)
- Native infilling/editing capability
- Bidirectional context for all tokens

## Implementation Notes

### Training Best Practices

1. **Start Small**: Always debug with "tiny" config on small dataset (10K samples, 1000 steps) before scaling up
2. **Gradient Accumulation**: If batch_size=64 doesn't fit in memory, use `grad_accum_steps=2` or more
3. **Learning Rate**: 3e-4 works well for small/medium models. Scale proportionally for larger models
4. **Warmup**: 1000 steps warmup prevents training instability
5. **Monitoring**: Watch masked accuracy (should reach 30-50% by end of training) and mask_rate (should average ~0.5)

### Common Issues

**Training loss not decreasing:**
- Check that masking is working (verify mask_rate > 0)
- Ensure learning rate warmup is enabled
- Try smaller model first to verify code correctness

**GPU OOM during training:**
- Reduce batch_size
- Use gradient accumulation to maintain effective batch size
- Switch to smaller model config
- Reduce max_seq_len

**Poor generation quality:**
- Train longer (50K steps minimum for coherent output)
- Increase model size (small → medium)
- Tune temperature (lower = more conservative, higher = more creative)
- Increase num_steps during sampling (25 → 50)

### Model Architecture Insights

**Why Bidirectional Attention?**
- Diffusion models need to predict based on both past and future context
- Unlike autoregressive models (causal masking), we see all unmasked tokens
- This allows the model to use more information for predictions

**Timestep Conditioning:**
- Timestep embeddings tell the model "how noisy is this input?"
- At t=1: input is all masks → model makes rough predictions
- At t=0: input is nearly clean → model makes fine-grained refinements
- Critical for multi-step denoising to work correctly

**Embedding Design Choices:**
- Currently using sinusoidal embeddings for timestep and position
- Alternative approaches worth experimenting with:
  - **Raw scalar + MLP**: Project t directly via learned linear layers
  - **ALiBi**: Add linear position biases directly to attention scores
  - **Learned lookup**: Discretize t into buckets with learned embeddings

**Output Projection:**
- Model outputs logits for ALL vocabulary positions at ALL sequence positions
- During training: only compute loss on masked positions
- During sampling: sample from distribution, but only update masked positions

### Code Organization

```
src/
├── core/           # Core model and diffusion implementations
│   ├── model.py              # DiffusionTransformer
│   ├── diffusion.py          # DiscreteDiffusion
│   ├── sparse_model.py       # SparseDenoiser, BilateralSparseDenoiser
│   └── sparse_diffusion.py   # SparseState, SparseDiffusion
├── training/       # Training scripts
│   ├── train.py              # Standard diffusion training
│   └── train_bilateral.py    # SDD training
├── generation/     # Generation scripts
│   └── generate.py           # Interactive generation
├── data/           # Data preparation
│   └── data_prep.py          # Dataset and tokenizer
└── evaluation/     # Evaluation and visualization
    └── evaluate.py           # Perplexity and generation eval
```

### Extension Ideas

1. **Continuous embedding diffusion**: Diffuse in embedding space instead of discrete tokens
2. **Constrained generation**: Add hard constraints during sampling
3. **Hierarchical diffusion**: Diffuse coarse structure first, then fine details
4. **Classifier-free guidance**: Train with and without conditioning for controllable generation

### Performance Benchmarking

**Quality Metrics:**
- Perplexity on held-out test set (target: < 30 for TinyStories)
- Masked token prediction accuracy (target: 30-50%)
- Manual inspection of generated samples for coherence

**Speed Metrics:**
- Training throughput: tokens/sec during training
- Generation latency: time for full sequence generation
- Compare against autoregressive baseline if available
