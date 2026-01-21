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

All models trained on TinyStories dataset, ~19M parameters, with matched hyperparameters for fair comparison.

### Quantitative Comparison

| Model | Train Loss | Val Loss | Accuracy | Notes |
|-------|------------|----------|----------|-------|
| **AR** | ~1.5 | ~1.8 | 56% | Coherent generation |
| Discrete Diffusion | ~2.5 | ~2.8 | 38% | Incoherent at this scale |
| **SDD** (10K steps) | 0.58 | 0.81 | 85% | High accuracy, repetitive generation |

### SDD Training Progress (Latest Run: Fixed k=4, 10K steps)

Standard training with fixed k=4 (no curriculum):

| Steps | Train Loss | Val Loss | Accuracy |
|-------|------------|----------|----------|
| 1,000 | 1.58 | 1.42 | 78% |
| 2,000 | 1.33 | 1.26 | 80% |
| 5,000 | 0.93 | 0.97 | 84% |
| 10,000 | 0.58 | 0.81 | 85% |

### Generation Quality

**AR** produces coherent TinyStories-style text:
```
Once upon a time, there was a little girl named Lily. She lived in a big house with her mommy and daddy...
```

**SDD** achieves strong training metrics but generation shows mode collapse:
```
Step 2000: Anna decided Ben is deep.. tired watched snake..... deep.....
Step 6000: Anna and Anna are, to the the. says a a a a a the a a a it,.
Step 10000: Dave loves... always always artist. the offered...ops...ops...
```

### What We've Tried

**Worked:**
- **Probability-based swapping**: Fixed train/inference mismatch where training saw different distribution than inference initialization
- **Simplified SparseState**: Removed redundant embeddings field (now looked up from indices)
- **Full L×k attention**: All candidates attend to all others across positions
- **Fair comparison infrastructure**: Matched hyperparameters (LR schedule, optimizer, batch size) across AR/MDLM/SDD

**Didn't Work:**
- **Two-step training** (COLING 2025 approach): Model predicts, re-noises prediction, predicts again. Failed completely - stuck at 7% accuracy. The model can't learn from its own bad predictions early in training (chicken-and-egg problem).
- **k curriculum (1→2→4→8)**: Intended to ease training, but fixed k=4 works just as well
- **Longer training**: 10K steps with 85% accuracy still produces repetitive text

### Conclusions

1. **SDD learns effectively** - 85% accuracy shows the model learns token distributions well
2. **Generation gap persists** - Good training metrics don't translate to coherent sampling
3. **The bottleneck is sampling, not training** - The iterative denoising causes mode collapse regardless of prediction accuracy
4. **Two-step training is a dead end** - Can't bootstrap from random initialization
5. **AR still wins for coherence** - At this scale, autoregressive remains more practical

### Open Questions

- Can improved sampling strategies (temperature scheduling, nucleus sampling) fix generation?
- Would classifier-free guidance or other steering techniques help?
- Is there a fundamental issue with discrete diffusion at small scale?

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

# SDD with experimental options
python src/training/train_bilateral.py --fixed_k 8            # Skip k curriculum
python src/training/train_bilateral.py --two_step_training    # Two-step training (experimental)

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
tests/              # Test suite (93% coverage)
```

## License

MIT
