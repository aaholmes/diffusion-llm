# LLM Training and Fine-tuning Pipeline

## Goal
Create a complete, architecture-agnostic pipeline for training and fine-tuning LLMs, starting small and scaling up. Support AR, Diffusion, and SDD architectures with unified metrics, W&B logging, and evaluation.

---

## Current State Analysis

### What Exists
| Component | AR | Diffusion | SDD |
|-----------|-----|-----------|-----|
| Training script | ✅ train_ar_text.py | ✅ train.py | ✅ train_bilateral.py |
| W&B integration | ❌ | ✅ | ❌ |
| Resume training | ❌ | ✅ | ❌ |
| CSV logging | ❌ | ✅ | ❌ |
| Evaluation | ✅ | ✅ | ✅ |
| Generation | ✅ | ✅ | ✅ |

### Gaps to Fill
1. **W&B logging** for AR and SDD trainers
2. **Instruction fine-tuning** scripts (none exist)
3. **OpenWebText/FineWeb** data preparation (only TinyStories exists)
4. **Unified evaluation** with consistent metrics across architectures
5. **Resume training** for AR and SDD

---

## Implementation Phases

### Phase 1: Standardize Training Infrastructure (Small Scale)

**Goal:** Ensure all 3 architectures have consistent logging, metrics, and W&B support.

#### 1.1 Add W&B to train_ar_text.py
**File:** `src/training/train_ar_text.py`

Add arguments:
```python
--no_wandb              # Disable W&B (enabled by default)
--wandb_project STR     # Default: "diffusion-lm"
--wandb_run_name STR    # Auto-generated if None
```

Metrics to log:
- `train/loss` - Cross-entropy loss
- `train/accuracy` - Token prediction accuracy
- `train/perplexity` - exp(loss)
- `train/lr` - Learning rate
- `train/tokens_per_sec` - Throughput
- `val/loss`, `val/accuracy`, `val/perplexity`

#### 1.2 Add W&B to train_bilateral.py
**File:** `src/training/train_bilateral.py`

Same arguments as above, plus:
- `train/k` - Current k value in curriculum
- `train/ignore_acc` - IGNORE token accuracy

#### 1.3 Add Resume Training
Both AR and SDD trainers need `--resume_from` argument that restores:
- Model weights
- Optimizer state
- Global step
- Best validation loss

#### 1.4 Standardize Checkpoint Format
All trainers should save:
```python
{
    "global_step": int,
    "model_state_dict": dict,
    "model_config": dict,
    "optimizer_state_dict": dict,
    "scaler_state_dict": dict,  # If using AMP
    "train_config": dict,
    "best_val_loss": float,
    "architecture": str,  # "ar", "diffusion", or "sdd"
}
```

---

### Phase 2: Unified Evaluation Script

**Goal:** Single script to evaluate any trained model with consistent metrics.

**File:** `scripts/evaluate_models.py`

#### Metrics for All Architectures:
| Metric | Description | How Computed |
|--------|-------------|--------------|
| **Perplexity** | Language modeling quality | exp(CE loss on val set) |
| **Accuracy** | Token prediction accuracy | Correct predictions / total |
| **Generation Quality** | Sample coherence | Human inspection + Self-BLEU |
| **Inference Speed** | Tokens per second | Benchmark generation |

#### CLI Interface:
```bash
python scripts/evaluate_models.py \
    --checkpoint PATH \
    --data_dir data_full \
    --architecture {ar,diffusion,sdd}  # Auto-detect if not specified
    --metrics perplexity,accuracy,generation,speed \
    --num_samples 100 \
    --output results.json
```

#### Auto-detection:
Read checkpoint's `architecture` field or infer from model config keys.

---

### Phase 3: Instruction Fine-tuning

**Goal:** Fine-tune pretrained models to follow instructions.

#### 3.1 Instruction Data Preparation
**File:** `src/data/prep_instruction_data.py`

Support datasets:
- **Alpaca** (52k examples) - Primary
- **Dolly-15k** (15k examples) - Alternative
- **Custom** (user-provided JSON)

Format:
```json
{
    "instruction": "Write a haiku about programming",
    "input": "",
    "output": "Silent keystrokes fall\nBugs emerge from tangled code\nCoffee grows cold now"
}
```

Tokenization format:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<EOS>
```

Output files:
```
data_instruct/
├── train_tokens.pt      # [N, max_seq_len]
├── train_masks.pt       # [N, max_seq_len] - 1 for response tokens, 0 otherwise
├── val_tokens.pt
├── val_masks.pt
├── tokenizer.json
└── config.json
```

#### 3.2 Fine-tuning Script
**File:** `src/training/finetune.py`

Key differences from pretraining:
- **Loss masking:** Only compute loss on response tokens
- **Lower learning rate:** 1e-5 to 5e-5 (vs 3e-4 for pretraining)
- **Fewer steps:** 1-3 epochs, typically 1-5k steps
- **No warmup or minimal warmup:** 100 steps

Arguments:
```bash
python src/training/finetune.py \
    --pretrained_checkpoint PATH \
    --data_dir data_instruct \
    --architecture {ar,diffusion,sdd} \
    --learning_rate 2e-5 \
    --max_steps 3000 \
    --eval_every 500 \
    --wandb_project diffusion-lm-finetune
```

#### 3.3 Instruction Evaluation
**File:** `scripts/evaluate_instruction.py`

Metrics:
- **Response quality:** Generate responses, compare to held-out test set
- **Instruction following:** Does output match instruction intent?
- **Format compliance:** Does output follow expected format?

---

### Phase 4: Scale Up Data (OpenWebText)

**Goal:** Support larger, more diverse datasets.

#### 4.1 OpenWebText Preparation
**File:** `src/data/prep_openwebtext.py`

Process:
1. Download OpenWebText (~8GB compressed)
2. Train larger BPE tokenizer (32k vocab)
3. Tokenize and chunk into sequences
4. Save as memory-mapped files for efficiency

Arguments:
```bash
python src/data/prep_openwebtext.py \
    --output_dir data_openwebtext \
    --vocab_size 32000 \
    --max_seq_len 1024 \
    --num_workers 8
```

Output:
```
data_openwebtext/
├── train_tokens.bin     # Memory-mapped for large datasets
├── val_tokens.bin
├── tokenizer.json
└── config.json
```

#### 4.2 Streaming DataLoader
For large datasets, use memory-mapped loading:
```python
class MemmapDataset(Dataset):
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
```

---

### Phase 5: Scale Up Models (300M)

**Goal:** Train larger models with same infrastructure.

#### Model Configurations:
| Config | d_model | n_heads | n_layers | d_ff | Params |
|--------|---------|---------|----------|------|--------|
| small | 384 | 6 | 6 | 1536 | ~25M |
| medium | 512 | 8 | 8 | 2048 | ~50M |
| large | 768 | 12 | 12 | 3072 | ~125M |
| **xl** | 1024 | 16 | 24 | 4096 | ~300M |

#### Memory Optimization for 300M:
- Gradient checkpointing: `--gradient_checkpointing`
- Gradient accumulation: `--grad_accum_steps 4`
- Mixed precision: Already supported via AMP

#### Training Script Updates:
Add to all trainers:
```python
--model_config {small,medium,large,xl}
--gradient_checkpointing  # Trade compute for memory
```

---

## Execution Order

### Small-Scale Validation (Do First)
1. ✅ Train AR 17M on TinyStories (10k steps) - DONE
2. ✅ Train Diffusion 17M on TinyStories (10k steps) - DONE
3. ✅ Train SDD 17M on TinyStories (10k steps) - DONE
4. ⬜ Add W&B to AR trainer
5. ⬜ Add W&B to SDD trainer
6. ⬜ Create unified evaluation script
7. ⬜ Run full comparison with W&B tracking

### Instruction Fine-tuning Validation
8. ⬜ Create instruction data prep script
9. ⬜ Prepare small Alpaca subset (1k examples)
10. ⬜ Create fine-tuning script
11. ⬜ Fine-tune best small model
12. ⬜ Evaluate instruction following

### Scale Up
13. ⬜ Create OpenWebText data prep
14. ⬜ Add gradient checkpointing
15. ⬜ Train 300M AR on OpenWebText
16. ⬜ Fine-tune 300M on full Alpaca

---

## Files to Create/Modify

### New Files:
| File | Purpose |
|------|---------|
| `src/data/prep_instruction_data.py` | Prepare Alpaca/Dolly data |
| `src/data/prep_openwebtext.py` | Prepare OpenWebText |
| `src/training/finetune.py` | Fine-tuning script |
| `scripts/evaluate_models.py` | Unified evaluation |
| `scripts/evaluate_instruction.py` | Instruction eval |

### Files to Modify:
| File | Changes |
|------|---------|
| `src/training/train_ar_text.py` | Add W&B, resume, standardize checkpoints |
| `src/training/train_bilateral.py` | Add W&B, resume, standardize checkpoints |
| `src/core/model.py` | Add gradient checkpointing, 'xl' config |

---

## Verification

### Phase 1 Complete When:
- [ ] All 3 trainers log to W&B with same metric names
- [ ] Can resume any training run from checkpoint
- [ ] `scripts/evaluate_models.py` works for all architectures

### Phase 2 Complete When:
- [ ] Can fine-tune AR model on 1k Alpaca examples
- [ ] Fine-tuned model follows simple instructions
- [ ] W&B shows fine-tuning metrics

### Phase 3 Complete When:
- [ ] OpenWebText prepared and training works
- [ ] 300M model trains without OOM
- [ ] Full pipeline: pretrain → finetune → evaluate

---

## W&B Dashboard Layout

### Pretraining Runs:
```
Project: diffusion-lm
├── ar-tinystories-small-v1
├── diffusion-tinystories-small-v1
├── sdd-tinystories-small-v1
├── ar-openwebtext-xl-v1
└── ...
```

### Metrics Panels:
1. **Loss curves** (train/loss, val/loss)
2. **Perplexity** (train/perplexity, val/perplexity)
3. **Accuracy** (train/accuracy, val/accuracy)
4. **Learning rate** (train/lr)
5. **Throughput** (train/tokens_per_sec)
6. **Architecture-specific** (SDD: train/k)

### Run Naming Convention:
```
{architecture}-{dataset}-{size}-v{version}
Examples:
  ar-tinystories-small-v1
  diffusion-openwebtext-xl-v2
  sdd-tinystories-medium-v1-fixed-k8
```
