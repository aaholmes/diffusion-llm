# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **discrete diffusion language model** from scratch, trained on desktop GPU (RTX 5060 Ti 16GB) and deployed to Jetson Orin Nano for edge inference. The project demonstrates:

- Bidirectional transformer architecture for discrete diffusion
- MDLM-style masking/unmasking diffusion process
- Custom CUDA kernels for Jetson optimization
- Full pipeline: tokenization → training → inference → edge deployment

Target model size: 25-80M parameters, trained on TinyStories dataset.

## Key Commands

### Environment Setup

**Desktop (Training):**
```bash
conda create -n diffusion-lm python=3.11
conda activate diffusion-lm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tokenizers wandb einops tqdm
```

**Jetson Orin Nano (Inference):**
```bash
# Use JetPack 6.0+ with PyTorch wheel from NVIDIA
pip install torch-*.whl  # From NVIDIA Jetson PyTorch releases
pip install tokenizers tqdm
```

### Data Preparation

```bash
python data_prep.py          # Download TinyStories, train tokenizer, tokenize dataset
```

**Outputs:**
- `tokenizer.json` - BPE tokenizer with 8192 vocab
- `tokenized_data.pt` - Tensor of tokenized sequences

### Training

```bash
python train.py              # Train model with default config (small, 25M params)
```

**Key parameters** (edit in `train.py` or pass as args):
- `model_config`: "tiny" (10M), "small" (25M), "medium" (50M), "large" (80M)
- `batch_size`: 64 (adjust based on GPU memory)
- `max_steps`: 20000 (prototype), 50000 (full training)
- `learning_rate`: 3e-4
- `use_wandb`: True (for experiment tracking)

**Training outputs:**
- `checkpoints/model_step{N}.pt` - Periodic checkpoints
- `checkpoints/model_final.pt` - Final trained model

### Generation (Desktop)

```bash
python generate.py           # Interactive text generation
```

### Model Export for Jetson

```bash
python export_jetson.py      # Export to FP16 TorchScript
python export_jetson.py --tensorrt  # Export to TensorRT (run on Jetson)
```

**Outputs:**
- `model_jetson.pt` - FP16 optimized model
- `model_jetson_config.json` - Model configuration
- `model.trt` - TensorRT engine (optional)

### Jetson Inference

```bash
# On Jetson Orin Nano
python inference_jetson.py   # Interactive inference with benchmarking
```

### Custom CUDA Kernels (Jetson)

```bash
# Build custom kernels for Jetson optimization
python setup_kernels.py build_ext --inplace
```

**Note:** Custom kernels require CUDA toolkit installed on Jetson (included in JetPack).

### Testing

```bash
# Test model architecture
python model.py

# Test diffusion process
python diffusion.py

# Benchmark inference speed
python evaluate.py
```

## Architecture

### Core Components

The codebase follows a modular structure:

**1. Data Pipeline** (`data_prep.py`)
- Downloads TinyStories dataset
- Trains BPE tokenizer (vocab_size=8192)
- Tokenizes and pads sequences to fixed length (default 256)
- Special tokens: `<PAD>` (0), `<BOS>` (1), `<EOS>` (2), `<MASK>` (3), `<UNK>` (4)

**2. Model Architecture** (`model.py`)
- **DiffusionTransformer**: Bidirectional transformer with timestep conditioning
  - Token embedding + positional embedding
  - Sinusoidal timestep embeddings (added to all positions)
  - N transformer blocks with **bidirectional attention** (no causal masking)
  - Output projection to vocabulary logits
- **Key difference from autoregressive LLMs**: Uses bidirectional attention, conditions on noise level `t`, predicts entire sequence at once

**3. Diffusion Process** (`diffusion.py`)
- **DiscreteDiffusion**: Implements discrete masking diffusion
  - **Forward process** (`q_sample`): Gradually mask tokens based on noise schedule
  - **Reverse process** (`p_sample`): Iteratively unmask tokens using model predictions
  - **Noise schedule**: Cosine or linear (cosine is smoother near endpoints)
  - **Training**: Cross-entropy loss only on masked positions
  - **Sampling**: Start from all masks, denoise over N steps (typically 25-50)

**4. Training** (`train.py`)
- Mixed precision training (FP16) with gradient accumulation
- AdamW optimizer with linear warmup + cosine decay
- Model compilation via `torch.compile()` for speed (PyTorch 2.0+)
- Wandb integration for experiment tracking
- Checkpointing every N steps

**5. Generation** (`generate.py`)
- Interactive text generation with temperature and top-k sampling
- Supports conditional generation (prompt-based)
- Multiple samples per prompt

**6. Jetson Optimization** (`export_jetson.py`, `inference_jetson.py`)
- FP16 conversion for 2x memory reduction
- TorchScript tracing for JIT optimization
- Optional TensorRT compilation for maximum speed
- Custom CUDA kernels for fused attention (advanced)

### Model Configurations

Defined in `model.py`:

| Config | Params | d_model | n_heads | n_layers | d_ff | VRAM (Training) | VRAM (Inference FP16) |
|--------|--------|---------|---------|----------|------|-----------------|----------------------|
| tiny   | ~10M   | 256     | 4       | 4        | 1024 | ~2GB           | ~0.5GB               |
| small  | ~25M   | 384     | 6       | 6        | 1536 | ~4GB           | ~1GB                 |
| medium | ~50M   | 512     | 8       | 8        | 2048 | ~6GB           | ~2GB                 |
| large  | ~80M   | 640     | 10      | 10       | 2560 | ~10GB          | ~3GB                 |

**Recommendation**: Start with "small" for prototyping, use "medium" for best quality/speed tradeoff on Jetson.

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

**Jetson inference too slow:**
- Verify FP16 conversion worked
- Try TensorRT compilation
- Reduce num_steps (25 → 10 for faster but lower quality)
- Reduce sequence length
- Profile with `nsys` to identify bottlenecks

### Jetson-Specific Notes

**Memory Constraints (8GB):**
- Use FP16 for all inference (reduces memory by 2x)
- Batch size = 1 for inference
- Max sequence length ≤ 128 recommended
- Clear CUDA cache between runs: `torch.cuda.empty_cache()`

**Performance Expectations (FP16, no custom kernels):**
- Small model (25M): 20-40 tokens/sec @ 64 length, 25 steps
- Medium model (50M): 10-20 tokens/sec @ 64 length, 25 steps
- Custom CUDA kernels can provide 2-3x speedup (requires advanced CUDA knowledge)

**TensorRT Compilation:**
- Only works for fixed input shapes
- Must compile on Jetson (not desktop) due to architecture differences
- Best for production deployment after model is finalized

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

**Output Projection:**
- Model outputs logits for ALL vocabulary positions at ALL sequence positions
- During training: only compute loss on masked positions
- During sampling: sample from distribution, but only update masked positions

### Code Organization Principles

When implementing new features:

1. **Model changes** go in `model.py` - keep architecture clean and modular
2. **Diffusion logic** goes in `diffusion.py` - separate from model definition
3. **Training infrastructure** goes in `train.py` - checkpointing, logging, etc.
4. **Inference optimizations** go in respective files (`generate.py`, `inference_jetson.py`)
5. **Keep configs in dictionaries** for easy experimentation (see MODEL_CONFIGS in `model.py`)

### Extension Ideas

Based on PROJECT_PLAN.md, potential extensions include:

1. **Multimodal conditioning**: Use Jetson camera + CLIP embeddings to condition generation on images
2. **Continuous embedding diffusion**: Diffuse in embedding space instead of discrete tokens (research direction)
3. **Constrained generation**: Add hard constraints during sampling (syntax checking, type correctness, etc.)
4. **Alternative noise schedules**: Experiment with different mask_rate functions
5. **Hierarchical diffusion**: Diffuse coarse structure first, then fine details

### Performance Benchmarking

When evaluating model performance:

**Quality Metrics:**
- Perplexity on held-out test set (target: < 30 for TinyStories)
- Masked token prediction accuracy (target: 30-50%)
- Manual inspection of generated samples for coherence

**Speed Metrics (Desktop):**
- Training throughput: tokens/sec during training
- Generation latency: time for full sequence generation
- Compare against autoregressive baseline if available

**Speed Metrics (Jetson):**
- Measure across different (seq_length, num_steps) configurations
- Report tokens/sec including all denoising steps
- Use `benchmark()` method in `inference_jetson.py`

### CUDA Kernel Development (Advanced)

If implementing custom kernels:

1. **Start with PyTorch reference implementation** - verify correctness first
2. **Profile to find bottlenecks** - use `torch.profiler` or `nsys`
3. **Focus on memory-bound operations** - attention is main target for optimization
4. **Jetson Orin architecture** - SM87, 1024 CUDA cores, 48KB shared memory per SM
5. **Test incrementally** - compare kernel output against PyTorch to catch bugs early
6. **Key optimizations**:
   - Use FP16 throughout (2x faster on Jetson)
   - Tile computations to fit in shared memory
   - Fuse operations (e.g., softmax + attention matmul)
   - Vectorized loads/stores where possible

Refer to FlashAttention paper and NVIDIA's CUTLASS library for kernel inspiration.
