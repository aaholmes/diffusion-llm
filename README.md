# Diffusion LLM

Exploring discrete diffusion architectures for text generation. Implements and compares three approaches: autoregressive (baseline), discrete diffusion (MDLM-style), and a novel Sparse Distribution Diffusion (SDD) approach.

## Architectures

| Model | Approach | Description |
|-------|----------|-------------|
| **Autoregressive (AR)** | Next-token prediction | Standard causal transformer baseline |
| **Discrete Diffusion** | Mask/unmask | MDLM-style: mask tokens, learn to unmask |
| **Sparse Distribution Diffusion** | k weighted candidates | Novel: each position holds k candidates with weights |

## Sparse Distribution Diffusion (SDD)

The novel contribution: instead of discrete tokens or full distributions, each position holds **k weighted candidates**:

```
Standard:  position = "cat"
SDD:       position = [(0.6, "cat"), (0.3, "dog"), (0.1, "pet")]
```

Mathematically: `position_i = Σ_{j=1}^{k} prob_ij × embedding_j`

**Key Design Choices:**

| Decision | Rationale |
|----------|-----------|
| k=8 candidates | Full vocab (8192) intractable; k=8 captures uncertainty tractably |
| Coarse-grained probs | Probs are true softmax values (sum < 1), not renormalized over k |
| 1/vocab_size at max noise | Represents true uncertainty over full vocabulary |
| Probability-based swapping | High-prob tokens rarely swapped, low-prob often swapped |
| No re-noising in sampling | DDIM-style deterministic; re-noising destroys predictions |
| Simplified state | SparseState stores (probs, indices); embeddings looked up when needed |

**Probability Semantics:** At maximum noise, each of k candidates has probability `1/vocab_size` (not `1/k`). This "coarse-grained" interpretation means we're sampling k tokens from a uniform distribution over the full vocabulary, preserving the true softmax semantics from `denoise_step`.

**Noise Injection (Probability-Based Swapping):**
```
Clean (t=0):    "cat": 97%,  "dog": 1%,  "pet": 1%,  "hat": 1%
Noisy (t=0.5):  "cat": 69%,  "dog": 0.7%, "???": 0.01%, "hat": 0.7%  (one swapped)
Noisy (t=1.0):  "???": 0.01%, "???": 0.01%, "???": 0.01%, "???": 0.01%  (all random)
```
- Swap probability depends on original token probability (before flattening)
- High-probability candidates (ground truth) are rarely swapped
- Swapped tokens get random indices with 1/vocab_size probability
- At t=1: training sees same random uniform distribution as inference start

**Architecture:** Full sparse attention preserving [B, L, k, D] throughout:
- **Full (L×k) attention**: Every candidate at every position attends to all others
- **Probability biasing**: High-probability candidates receive more attention
- **Learned readout**: Soft attention over k → single output per position

## Results

All models trained on TinyStories dataset, ~19M parameters.

### Quantitative Comparison

| Model | Train Loss | Val Loss | Accuracy | Notes |
|-------|------------|----------|----------|-------|
| **AR** | ~1.5 | ~1.8 | 56% | Coherent generation |
| Discrete Diffusion | ~2.5 | ~2.8 | 38% | Incoherent at this scale |
| **SDD** (5K steps) | 0.87 | 1.00 | 85% | High accuracy, repetitive generation |

### SDD Training Progress (Latest Run)

With probability-based swapping and simplified SparseState:

| Steps | Train Loss | Val Loss | Accuracy |
|-------|------------|----------|----------|
| 1,000 | 2.2 | 1.5 | 62% |
| 2,000 | 1.6 | 1.2 | 74% |
| 3,000 | 1.4 | 1.1 | 77% |
| 5,000 | 0.87 | 1.0 | 85% |

### Generation Quality

**AR** produces coherent TinyStories-style text:
```
Once upon a time, there was a little girl named Lily. She lived in a big house with her mommy and daddy...
```

**SDD** achieves strong training metrics but generation remains repetitive:
```
Tom and a were were in the....
Tom they and the to in the....
```

### What We've Tried

- **Probability-based swapping**: Fixed train/inference mismatch where training saw different distribution than inference initialization
- **Simplified SparseState**: Removed redundant embeddings field (now looked up from indices)
- **Curriculum learning** (k=1→2→4→8): Helps early training stability
- **Full L×k attention**: All candidates attend to all others across positions

### Conclusions

1. **SDD learns effectively** - 85% accuracy shows the model learns token distributions well
2. **Generation gap persists** - Good training metrics don't translate to coherent sampling
3. **The bottleneck is sampling** - The iterative denoising produces repetitive text despite learning
4. **AR still wins for coherence** - At this scale, autoregressive remains more practical

### Open Questions

- Can improved sampling strategies (temperature scheduling, nucleus sampling) fix generation?
- Does longer training (50K+ steps) improve coherence?
- Would guidance techniques help steer generation?

## Quick Start

```bash
# Setup
pip install torch transformers datasets tokenizers tqdm wandb

# Prepare data (downloads TinyStories, trains tokenizer)
python src/data/data_prep.py

# Train (pick one)
python src/training/train_ar_text.py --max_steps 10000        # Autoregressive
python src/training/train.py --max_steps 10000                 # Discrete Diffusion
python src/training/train_bilateral.py --max_steps 10000       # SDD

# Evaluate and compare
python scripts/evaluate_comparison.py --benchmark
```

## Project Structure

```
src/
├── core/           # Model architectures
│   ├── model.py              # AR and Diffusion transformers
│   ├── diffusion.py          # Discrete diffusion process
│   ├── sparse_model.py       # SDD full sparse attention model
│   └── sparse_diffusion.py   # SDD noise/denoise process
├── training/       # Training scripts for each architecture
├── data/           # Data preparation (TinyStories)
└── evaluation/     # Metrics and visualization
scripts/            # Evaluation and comparison scripts
tests/              # Test suite (94% coverage)
```

## License

MIT
