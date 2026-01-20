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
| Two noise mechanisms | Prob flattening + embedding noise (no index swapping) |
| No re-noising in sampling | DDIM-style deterministic; re-noising destroys predictions |

**Probability Semantics:** At maximum noise, each of k candidates has probability `1/vocab_size` (not `1/k`). This "coarse-grained" interpretation means we're sampling k tokens from a uniform distribution over the full vocabulary, preserving the true softmax semantics from `denoise_step`.

**Why no index swapping?** Earlier versions swapped candidate indices with random tokens at high noise levels. This corrupted the signal—at high t, the model saw random indices with flat probabilities, making recovery impossible. By keeping indices fixed and only flattening probabilities + adding embedding noise, the model can learn to extract signal from uncertain distributions where indices still carry meaning.

**Architecture:** Full sparse attention preserving [B, L, k, D] throughout:
- **Full (L×k) attention**: Every candidate at every position attends to all others
- **Probability biasing**: High-probability candidates receive more attention
- **Learned readout**: Soft attention over k → single output per position

## Preliminary Results

All models trained on TinyStories dataset, ~17M parameters, 10k steps.

### Quantitative Comparison

| Model | Perplexity | Accuracy | Gen Time (64 tok) |
|-------|------------|----------|-------------------|
| **AR** | **6.0** | **56.0%** | 70ms |
| Discrete Diffusion | 27.0 | 38.3% | 59ms |
| SDD | 17.8 | 54.4% | 162ms |

### Generation Quality

**AR** produces coherent TinyStories-style text:
```
Once upon a time, there was a little girl named Lily. She lived in a big house with her mommy and daddy...
```

**Discrete Diffusion** produces incoherent output at this training scale.

**SDD** achieves decent training metrics but exhibits repetitive generation:
```
They They They obedient go delicious lived. and another and kite kite doors led delicious...
```

### What We Tried

- **Curriculum learning for SDD** (k=1→2→4→8 over training): No improvement over fixed k=8. Both achieve similar metrics but with different failure modes in generation.
- **Different k values**: Fixed k=8 works as well as curriculum, simpler is better.

### Conclusions

1. **AR wins at this scale** - 3x better perplexity than SDD, coherent generation
2. **SDD beats discrete diffusion on metrics** - Better perplexity (18 vs 27) and accuracy (54% vs 38%)
3. **SDD has a generation problem** - Good training metrics don't translate to coherent sampling. The iterative denoising process produces repetitive text despite the model learning reasonable token distributions.
4. **The bottleneck is sampling, not learning** - SDD's 54% accuracy suggests it learns useful representations, but the sampling algorithm needs work.

### Open Questions

- Can improved sampling strategies (temperature, scheduling) fix SDD generation?
- Does the gap narrow at larger scale or longer training?
- Would different noise injection approaches help?

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
