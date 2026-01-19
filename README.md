# Sparse Distribution Diffusion for Text

A novel approach to discrete diffusion that operates on **sparse probability distributions** in embedding space rather than discrete tokens or full dense distributions.

## The Core Idea

Traditional diffusion approaches for text fall into two categories:

| Approach | Representation | Example |
|----------|---------------|---------|
| **Discrete** | Single token per position | D3PM, MDLM |
| **Continuous (Dense)** | Full distribution over V tokens | Diffusion-LM |

**Sparse Distribution Diffusion (SDD)** introduces a novel middle ground:

```
position_i = Σ_{j=1}^{k} prob_ij × embedding_j
```

Each position is represented as a **weighted mixture of k embeddings** (typically k=8), which is mathematically a sum of k Dirac delta functions in continuous embedding space.

### Why This Matters

1. **vs Discrete (D3PM, MDLM)**: We maintain probability distributions, enabling smooth transitions during denoising instead of abrupt token swaps

2. **vs Dense (Diffusion-LM)**: We're sparse (k terms), not dense (V terms), making computation tractable while preserving expressiveness

3. **Smooth denoising**: Probabilities shift gradually between candidates rather than hard token replacements

4. **Maintains uncertainty**: A position can represent mixtures of meanings (e.g., "big"/"large" both plausible) until later steps

5. **Efficient**: Track k candidates per position, not the full vocabulary

### Visual Comparison

```
Discrete Diffusion (D3PM/MDLM):
  Position: "cat" → "cat" → "dog" → "the"  (abrupt jumps)

Dense Diffusion (Diffusion-LM):
  Position: [0.001, 0.002, ..., 0.003]  (V=50k probabilities)

Sparse Distribution Diffusion:
  Position: [(0.6, "cat"), (0.3, "dog"), (0.1, "pet")]  (k=3 candidates)
           → [(0.8, "the"), (0.15, "a"), (0.05, "this")]  (smooth refinement)
```

## Architecture: Bilateral Sparse Attention

The **BilateralSparseDenoiser** (v2) preserves the full `[batch, seq_len, k, d_model]` tensor throughout all transformer layers, enabling rich interactions between candidates.

### Key Innovation: Two Types of Attention

**Intra-Position Attention** (k×k per position):
- Candidates within each position attend to each other
- Probability bias: attend more to high-probability candidates
- Enables competition and refinement within positions

**Inter-Position Attention** (L×L across positions):
- Pool each position's candidates (probability-weighted)
- Standard self-attention across sequence positions
- Broadcast context back to all candidates
- Enables context propagation across the sequence

```
Input: SparseState [B, L, k, E]
    ↓
Probability encoding + position/time embeddings
    ↓
N × BilateralSparseBlock:
    ├── IntraPositionAttn (k×k, with prob bias)
    ├── InterPositionAttn (L×L, pooled)
    ├── CrossAttn (for conditioning)
    └── FFN
    ↓
Learned readout (collapse k → 1)
    ↓
Output logits [B, L, vocab_size]
```

## Quick Start

### Prerequisites

```bash
pip install torch transformers datasets tokenizers tqdm
```

### Data Preparation

```bash
# Download TinyStories and prepare tokenized data
python data_prep.py
```

### Training

```bash
# Train with BilateralSparseDenoiser (v2, recommended)
python train_bilateral.py --max_steps 20000

# Compare with SparseDenoiser (v1, simpler baseline)
python train_bilateral.py --model_version v1 --max_steps 20000
```

### Key Training Options

```bash
python train_bilateral.py \
    --data_dir data_full \
    --checkpoint_dir checkpoints_sdd \
    --model_version v2 \
    --d_model 384 \
    --n_layers 6 \
    --n_heads 6 \
    --k_schedule 1,2,4,8 \
    --max_steps 20000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model_version` | v2 | v1=SparseDenoiser, v2=BilateralSparseDenoiser |
| `--k_schedule` | 1,2,4,8 | Curriculum for increasing k during training |
| `--d_model` | 384 | Transformer hidden dimension |
| `--n_layers` | 6 | Number of transformer layers |

## The SparseState Representation

The core data structure is `SparseState`:

```python
class SparseState:
    probs: Tensor   # [batch, seq_len, k] - probabilities
    embeds: Tensor  # [batch, seq_len, k, embed_dim] - embeddings
    indices: Tensor # [batch, seq_len, k] - token indices
```

### Creating a SparseState

```python
from sparse_diffusion import SparseDiffusion, SDDConfig

# From discrete tokens (for training)
config = SDDConfig(vocab_size=8192, k=8)
diffusion = SparseDiffusion(config, embedding_table)
state = diffusion.state_from_tokens(tokens)

# Random initialization (for generation)
state = diffusion.initialize_state(batch_size=4, seq_len=64, device="cuda")
```

### The Diffusion Process

**Forward Process (Noise Injection):**
1. Mix probabilities toward uniform (flatten distribution)
2. Add Gaussian noise to embeddings (blur meanings)
3. Swap some embeddings with random ones (inject new candidates)

**Reverse Process (Denoising):**
1. Model predicts logits over full vocabulary
2. Convert to top-k sparse state (renormalize probabilities)
3. Repeat with decreasing noise level

```python
# Sampling
tokens, final_state = diffusion.sample(
    model,
    batch_size=4,
    seq_len=64,
    num_steps=25,
    device="cuda",
)
```

## Model Configurations

| Config | d_model | Heads | Layers | k | Parameters |
|--------|---------|-------|--------|---|------------|
| tiny   | 256     | 4     | 4      | 8 | ~6M        |
| small  | 384     | 6     | 6      | 8 | ~20M       |
| medium | 512     | 8     | 8      | 8 | ~45M       |
| large  | 768     | 12    | 12     | 8 | ~100M      |

## Key Components

### sparse_diffusion.py

Core diffusion logic:
- `SparseState`: The (probs, embeds, indices) representation
- `SparseDiffusion`: Noise injection and sampling utilities
- `SDDConfig`: Configuration for the diffusion process

### sparse_model.py

Denoiser architectures:
- `SparseDenoiser` (v1): Aggregates k→1 before transformer
- `BilateralSparseDenoiser` (v2): Maintains k throughout with bilateral attention
- `IntraPositionAttention`: k×k attention within positions
- `InterPositionAttention`: L×L attention across positions

### train_bilateral.py

Training script with:
- k-curriculum (gradually increase k during training)
- Mixed precision (AMP)
- Warmup + cosine decay learning rate
- Periodic generation samples

## Research Context

### Related Work

**Discrete Diffusion:**
- [D3PM](https://arxiv.org/abs/2107.03006) - Discrete denoising diffusion
- [MDLM](https://arxiv.org/abs/2406.07524) - Masked diffusion language model

**Continuous Diffusion for Text:**
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - Embedding space diffusion

**Diffusion Image Captioning:**
- [DDCap](https://arxiv.org/abs/2211.11694) - Discrete diffusion captioning
- [LaDiC](https://arxiv.org/abs/2404.10763) - Latent diffusion captioner

### Novelty of SDD

1. **Sparse representation**: k weighted embeddings instead of single token or full distribution
2. **Bilateral attention**: Both within-position (k×k) and across-position (L×L) attention
3. **Probability bias**: Attention biased toward high-probability candidates
4. **k-curriculum**: Gradually increase expressiveness during training

## Extended Features

### Conditioning (Image Captioning)

The model supports cross-attention conditioning for multimodal tasks:

```python
model, sdd_config = create_bilateral_sparse_model(
    vocab_size=8192,
    encoder_dim=768,  # CLIP feature dimension
)

# During forward pass
logits = model(state, t, encoder_output=clip_features)
```

### Variable-Length Generation

Uses an IGNORE token to enable variable-length sequences:
- During training: PAD tokens are replaced with IGNORE as targets
- During generation: IGNORE tokens are filtered from output
- Noise-aware weighting: IGNORE loss weight increases as t→0

## Project Structure

```
diffusion-llm/
├── sparse_diffusion.py      # Core SDD: SparseState, SparseDiffusion
├── sparse_model.py          # Models: SparseDenoiser, BilateralSparseDenoiser
├── train_bilateral.py       # Training script for SDD
├── data_prep.py             # TinyStories data preparation
├── model.py                 # Baseline discrete diffusion model
├── diffusion.py             # Baseline discrete diffusion process
├── train.py                 # Baseline training script
└── test_*.py                # Test suites
```

## Testing

```bash
# Run model tests
python sparse_model.py  # Built-in tests

# Run full test suite
pytest -v
```

## License

MIT

## Citation

```bibtex
@software{sparse_distribution_diffusion_2026,
  author = {Adam A Holmes},
  title = {Sparse Distribution Diffusion for Text Generation},
  year = {2026},
  url = {https://github.com/aaholmes/diffusion-llm}
}
```
