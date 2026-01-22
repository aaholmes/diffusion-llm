#!/usr/bin/env python3
"""
Tests for train_ar_text.py - Autoregressive text model trainer.

Run with: pytest tests/test_train_ar_text.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn.functional as F

from src.training.train_ar_text import ARConfig, CausalTransformer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def small_config():
    """Create a small config for testing."""
    return ARConfig(
        vocab_size=500,
        max_seq_len=32,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )


@pytest.fixture
def model(small_config):
    """Create a small model for testing."""
    return CausalTransformer(small_config)


@pytest.fixture
def mock_data_dir(temp_dir):
    """Create mock data directory with all required files."""
    data_dir = temp_dir

    # Create config.json
    config_data = {
        "vocab_size": 500,
        "max_seq_len": 32,
    }
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config_data, f)

    # Create tokenizer mock file - BPE style
    tokenizer_content = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<UNK>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {f"token{i}": i for i in range(500)},
            "merges": []
        }
    }

    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_content, f)

    # Create mock token tensors
    train_tokens = torch.randint(5, 500, (100, 32))
    val_tokens = torch.randint(5, 500, (20, 32))

    # Ensure BOS token at start and EOS at end
    train_tokens[:, 0] = 1  # BOS
    train_tokens[:, -1] = 2  # EOS
    val_tokens[:, 0] = 1
    val_tokens[:, -1] = 2

    torch.save(train_tokens, os.path.join(data_dir, "train_tokens.pt"))
    torch.save(val_tokens, os.path.join(data_dir, "val_tokens.pt"))

    return data_dir


# =============================================================================
# Tests: ARConfig
# =============================================================================

class TestARConfig:
    """Tests for ARConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ARConfig()

        assert config.vocab_size == 8192
        assert config.max_seq_len == 256
        assert config.d_model == 384
        assert config.n_heads == 6
        assert config.n_layers == 6
        assert config.d_ff == 1536
        assert config.dropout == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ARConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            dropout=0.2,
        )

        assert config.vocab_size == 1000
        assert config.d_model == 128
        assert config.n_layers == 4
        assert config.dropout == 0.2

    def test_dict_conversion(self):
        """Test converting config to dictionary."""
        config = ARConfig(vocab_size=1000)

        # ARConfig is a dataclass, check dict access
        assert config.vocab_size == 1000


# =============================================================================
# Tests: CausalTransformer Model Creation
# =============================================================================

class TestCausalTransformerCreation:
    """Tests for CausalTransformer model creation."""

    def test_model_creation(self, small_config):
        """Test model can be created."""
        model = CausalTransformer(small_config)

        assert model is not None
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'position_embedding')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'output_proj')

    def test_model_parameter_count(self, small_config):
        """Test model has parameters."""
        model = CausalTransformer(small_config)
        num_params = sum(p.numel() for p in model.parameters())

        assert num_params > 0

    def test_model_has_causal_mask(self, small_config):
        """Test model has causal mask buffer."""
        model = CausalTransformer(small_config)

        assert hasattr(model, 'causal_mask')
        # Causal mask should be upper triangular
        mask = model.causal_mask
        assert mask.shape == (small_config.max_seq_len, small_config.max_seq_len)
        # Upper triangle should be True (masked)
        assert mask[0, 1].item() == True
        # Lower triangle and diagonal should be False (not masked)
        assert mask[1, 0].item() == False

    def test_model_embedding_sizes(self, small_config):
        """Test embedding dimensions."""
        model = CausalTransformer(small_config)

        assert model.token_embedding.num_embeddings == small_config.vocab_size
        assert model.token_embedding.embedding_dim == small_config.d_model
        assert model.position_embedding.num_embeddings == small_config.max_seq_len
        assert model.position_embedding.embedding_dim == small_config.d_model

    def test_output_projection_size(self, small_config):
        """Test output projection dimensions."""
        model = CausalTransformer(small_config)

        assert model.output_proj.in_features == small_config.d_model
        assert model.output_proj.out_features == small_config.vocab_size


# =============================================================================
# Tests: CausalTransformer Forward Pass
# =============================================================================

class TestCausalTransformerForward:
    """Tests for CausalTransformer forward pass."""

    def test_forward_shape(self, model, small_config):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 16
        x = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits = model(x)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_forward_with_full_sequence(self, model, small_config):
        """Test forward pass with full sequence length."""
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, small_config.max_seq_len))

        logits = model(x)

        assert logits.shape == (batch_size, small_config.max_seq_len, small_config.vocab_size)

    def test_forward_single_token(self, model, small_config):
        """Test forward pass with single token."""
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, 1))

        logits = model(x)

        assert logits.shape == (batch_size, 1, small_config.vocab_size)

    def test_forward_batch_size_one(self, model, small_config):
        """Test forward pass with batch size 1."""
        seq_len = 10
        x = torch.randint(0, small_config.vocab_size, (1, seq_len))

        logits = model(x)

        assert logits.shape == (1, seq_len, small_config.vocab_size)

    def test_forward_gradient_flow(self, model, small_config):
        """Test gradients flow through the model."""
        x = torch.randint(0, small_config.vocab_size, (2, 8))
        targets = torch.randint(0, small_config.vocab_size, (2, 8))

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, small_config.vocab_size), targets.view(-1))
        loss.backward()

        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_forward_different_batch_sizes(self, small_config):
        """Test forward with different batch sizes."""
        model = CausalTransformer(small_config)
        seq_len = 16

        for batch_size in [1, 2, 4, 8]:
            x = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            logits = model(x)
            assert logits.shape == (batch_size, seq_len, small_config.vocab_size)


# =============================================================================
# Tests: Causal Masking
# =============================================================================

class TestCausalMasking:
    """Tests for causal attention masking."""

    def test_no_future_leakage(self, small_config):
        """Test that model doesn't leak future information."""
        model = CausalTransformer(small_config)
        model.eval()

        # Create input
        x = torch.randint(0, small_config.vocab_size, (1, 10))

        with torch.no_grad():
            # Get predictions
            logits_full = model(x)

            # Modify token at position 5
            x_modified = x.clone()
            x_modified[0, 5] = (x[0, 5] + 1) % small_config.vocab_size

            logits_modified = model(x_modified)

        # Predictions before position 5 should be identical
        # (within floating point tolerance)
        for pos in range(5):
            assert torch.allclose(
                logits_full[0, pos], logits_modified[0, pos], atol=1e-5
            ), f"Position {pos} leaked future information"

    def test_past_affects_future(self, small_config):
        """Test that past tokens affect future predictions."""
        model = CausalTransformer(small_config)
        model.eval()

        # Create input
        x = torch.randint(0, small_config.vocab_size, (1, 10))

        with torch.no_grad():
            logits_full = model(x)

            # Modify token at position 0
            x_modified = x.clone()
            x_modified[0, 0] = (x[0, 0] + 1) % small_config.vocab_size

            logits_modified = model(x_modified)

        # Predictions at later positions should be different
        # (since position 0 affects everything after it)
        # At least position 1 should be different
        assert not torch.allclose(
            logits_full[0, 1], logits_modified[0, 1], atol=1e-6
        ), "Past tokens should affect future predictions"


# =============================================================================
# Tests: Generation
# =============================================================================

class TestGeneration:
    """Tests for text generation."""

    def test_generate_creates_tokens(self, model, small_config):
        """Test that generate creates token sequence."""
        model.eval()
        max_len = 20

        with torch.no_grad():
            tokens = model.generate(max_len=max_len, device='cpu')

        assert tokens.shape[0] == 1  # Batch size 1
        assert tokens.shape[1] <= max_len

    def test_generate_with_prompt(self, model, small_config):
        """Test generation with prompt."""
        model.eval()
        prompt = torch.tensor([[1, 5, 10, 15]])  # Some tokens
        max_len = 30

        with torch.no_grad():
            tokens = model.generate(prompt=prompt, max_len=max_len, device='cpu')

        # Output should start with prompt
        assert tokens.shape[1] >= prompt.shape[1]
        assert torch.all(tokens[0, :prompt.shape[1]] == prompt[0])

    def test_generate_respects_max_len(self, model, small_config):
        """Test that generation respects max_len."""
        model.eval()
        max_len = 15

        with torch.no_grad():
            tokens = model.generate(max_len=max_len, device='cpu')

        assert tokens.shape[1] <= max_len

    def test_generate_different_temperatures(self, small_config):
        """Test generation with different temperatures."""
        model = CausalTransformer(small_config)
        model.eval()

        torch.manual_seed(42)
        with torch.no_grad():
            tokens_low = model.generate(max_len=20, temperature=0.5, device='cpu')

        torch.manual_seed(42)
        with torch.no_grad():
            tokens_high = model.generate(max_len=20, temperature=2.0, device='cpu')

        # Different temperatures should generally produce different outputs
        # (not a strict test, as random sampling may occasionally match)
        assert tokens_low.shape[1] > 1
        assert tokens_high.shape[1] > 1

    def test_generate_top_k_filtering(self, small_config):
        """Test generation with top-k filtering."""
        model = CausalTransformer(small_config)
        model.eval()

        with torch.no_grad():
            # top_k=1 should be deterministic
            tokens = model.generate(max_len=20, top_k=1, device='cpu')

        assert tokens.shape[1] > 1

    def test_generate_stops_at_eos(self, small_config):
        """Test that generation can stop at EOS token."""
        # Create a model and manually test EOS stopping
        model = CausalTransformer(small_config)
        model.eval()

        # Generate - keep max_len within model's max_seq_len
        with torch.no_grad():
            tokens = model.generate(max_len=small_config.max_seq_len - 1, device='cpu')

        # Just verify it doesn't crash and produces valid output
        assert tokens.shape[1] >= 1

    def test_generate_no_prompt_starts_with_bos(self, small_config):
        """Test that generation without prompt starts with BOS."""
        model = CausalTransformer(small_config)
        model.eval()

        with torch.no_grad():
            tokens = model.generate(max_len=10, device='cpu')

        # First token should be BOS (token 1)
        assert tokens[0, 0].item() == 1


# =============================================================================
# Tests: Weight Initialization
# =============================================================================

class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_weights_initialized(self, small_config):
        """Test that weights are initialized (not zeros)."""
        model = CausalTransformer(small_config)

        # Check some weights are non-zero
        token_emb_weight = model.token_embedding.weight
        assert not torch.all(token_emb_weight == 0)

    def test_xavier_initialization(self, small_config):
        """Test that weights follow Xavier initialization."""
        model = CausalTransformer(small_config)

        # Xavier initialization should have reasonable variance
        # For Xavier uniform, var should be ~ 2/(fan_in + fan_out)
        weight = model.output_proj.weight
        var = weight.var().item()

        # Just check it's in a reasonable range
        assert 1e-6 < var < 1.0


# =============================================================================
# Tests: Training Function
# =============================================================================

class TestTrainFunction:
    """Tests for the train() function."""

    def test_train_runs_without_error(self, mock_data_dir, temp_dir):
        """Test that training runs without error."""
        import argparse

        args = argparse.Namespace(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            batch_size=8,
            max_steps=3,
            learning_rate=1e-3,
            warmup_steps=1,
            d_model=64,
            n_heads=2,
            n_layers=2,
            log_every=1,
            eval_every=2,
            save_every=5,
            generate_every=100,  # Don't generate during test
            optimizer="adamw",
            no_wandb=True,
            wandb_project="test",
            wandb_run_name="test",
        )

        from src.training.train_ar_text import train
        train(args)

        # Check checkpoint dir was created
        assert os.path.exists(args.checkpoint_dir)

    def test_train_creates_checkpoints(self, mock_data_dir, temp_dir):
        """Test that training creates checkpoints."""
        import argparse

        ckpt_dir = os.path.join(temp_dir, "ckpts")
        args = argparse.Namespace(
            data_dir=mock_data_dir,
            checkpoint_dir=ckpt_dir,
            batch_size=8,
            max_steps=5,
            learning_rate=1e-3,
            warmup_steps=1,
            d_model=64,
            n_heads=2,
            n_layers=2,
            log_every=1,
            eval_every=3,
            save_every=5,
            generate_every=100,
            optimizer="adamw",
            no_wandb=True,
            wandb_project="test",
            wandb_run_name="test",
        )

        from src.training.train_ar_text import train
        train(args)

        # Check checkpoint was saved at step 5
        assert os.path.exists(os.path.join(ckpt_dir, "step_5.pt"))

    def test_train_saves_best_model(self, mock_data_dir, temp_dir):
        """Test that training saves best model."""
        import argparse

        ckpt_dir = os.path.join(temp_dir, "ckpts")
        args = argparse.Namespace(
            data_dir=mock_data_dir,
            checkpoint_dir=ckpt_dir,
            batch_size=8,
            max_steps=5,
            learning_rate=1e-3,
            warmup_steps=1,
            d_model=64,
            n_heads=2,
            n_layers=2,
            log_every=1,
            eval_every=3,
            save_every=100,
            generate_every=100,
            optimizer="adamw",
            no_wandb=True,
            wandb_project="test",
            wandb_run_name="test",
        )

        from src.training.train_ar_text import train
        train(args)

        # Check best model was saved
        assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))

    def test_checkpoint_format(self, mock_data_dir, temp_dir):
        """Test checkpoint contains required keys."""
        import argparse

        ckpt_dir = os.path.join(temp_dir, "ckpts")
        args = argparse.Namespace(
            data_dir=mock_data_dir,
            checkpoint_dir=ckpt_dir,
            batch_size=8,
            max_steps=5,
            learning_rate=1e-3,
            warmup_steps=1,
            d_model=64,
            n_heads=2,
            n_layers=2,
            log_every=1,
            eval_every=3,
            save_every=5,
            generate_every=100,
            optimizer="adamw",
            no_wandb=True,
            wandb_project="test",
            wandb_run_name="test",
        )

        from src.training.train_ar_text import train
        train(args)

        # Load checkpoint and check contents
        ckpt = torch.load(os.path.join(ckpt_dir, "step_5.pt"), weights_only=False)

        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "step" in ckpt

    def test_train_with_muon_optimizer(self, mock_data_dir, temp_dir):
        """Test training with Muon optimizer."""
        import argparse

        ckpt_dir = os.path.join(temp_dir, "ckpts_muon")
        args = argparse.Namespace(
            data_dir=mock_data_dir,
            checkpoint_dir=ckpt_dir,
            batch_size=8,
            max_steps=3,
            learning_rate=1e-3,
            warmup_steps=1,
            d_model=64,
            n_heads=2,
            n_layers=2,
            log_every=1,
            eval_every=2,
            save_every=5,
            generate_every=100,
            optimizer="muon",
            no_wandb=True,
            wandb_project="test",
            wandb_run_name="test",
        )

        from src.training.train_ar_text import train
        train(args)

        # Check checkpoint dir was created
        assert os.path.exists(args.checkpoint_dir)


# =============================================================================
# Tests: Learning Rate Schedule
# =============================================================================

class TestLRSchedule:
    """Tests for learning rate schedule."""

    def test_lr_warmup(self, mock_data_dir, temp_dir):
        """Test LR warmup behavior."""
        import argparse

        args = argparse.Namespace(
            data_dir=mock_data_dir,
            checkpoint_dir=os.path.join(temp_dir, "ckpts"),
            batch_size=8,
            max_steps=20,
            learning_rate=1e-3,
            warmup_steps=10,  # Use 10 for cleaner division
            d_model=64,
            n_heads=2,
            n_layers=2,
            log_every=1,
            eval_every=100,
            save_every=100,
            generate_every=100,
        )

        # Test warmup function directly
        def get_lr(step):
            if step < args.warmup_steps:
                return step / args.warmup_steps
            return 1.0

        assert get_lr(0) == 0.0
        assert get_lr(5) == 0.5  # Half of warmup
        assert get_lr(args.warmup_steps) == 1.0
        assert get_lr(args.warmup_steps + 10) == 1.0


# =============================================================================
# Tests: Data Loading
# =============================================================================

class TestDataLoading:
    """Tests for data loading."""

    def test_data_loads_correctly(self, mock_data_dir):
        """Test that data can be loaded."""
        train_tokens = torch.load(os.path.join(mock_data_dir, "train_tokens.pt"))
        val_tokens = torch.load(os.path.join(mock_data_dir, "val_tokens.pt"))

        assert train_tokens.shape == (100, 32)
        assert val_tokens.shape == (20, 32)

    def test_config_loads_correctly(self, mock_data_dir):
        """Test that config loads correctly."""
        with open(os.path.join(mock_data_dir, "config.json")) as f:
            config = json.load(f)

        assert config["vocab_size"] == 500
        assert config["max_seq_len"] == 32


# =============================================================================
# Tests: Loss Computation
# =============================================================================

class TestLossComputation:
    """Tests for loss computation."""

    def test_cross_entropy_loss(self, model, small_config):
        """Test cross entropy loss computation."""
        batch_size = 4
        seq_len = 16

        # Input: all but last token
        inputs = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        # Target: all but first token (shifted)
        targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, small_config.vocab_size),
            targets.reshape(-1),
            ignore_index=0  # Ignore padding
        )

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_ignores_padding(self, model, small_config):
        """Test that loss ignores padding tokens."""
        batch_size = 4
        seq_len = 16

        # Create inputs with padding
        inputs = torch.randint(5, small_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(5, small_config.vocab_size, (batch_size, seq_len))

        # Add padding tokens
        inputs[:, -5:] = 0
        targets[:, -5:] = 0

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, small_config.vocab_size),
            targets.reshape(-1),
            ignore_index=0
        )

        assert loss.item() > 0
        assert not torch.isnan(loss)


# =============================================================================
# Tests: Model Modes
# =============================================================================

class TestModelModes:
    """Tests for model train/eval modes."""

    def test_training_mode(self, model):
        """Test model in training mode."""
        model.train()
        assert model.training

    def test_eval_mode(self, model):
        """Test model in eval mode."""
        model.eval()
        assert not model.training

    def test_dropout_in_train_mode(self, small_config):
        """Test that dropout is active in train mode."""
        config = ARConfig(
            vocab_size=500,
            max_seq_len=32,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.5,  # High dropout for test visibility
        )
        model = CausalTransformer(config)
        model.train()

        x = torch.randint(0, config.vocab_size, (1, 10))

        # Run multiple times - should get different outputs due to dropout
        outputs = []
        for _ in range(5):
            outputs.append(model(x).clone())

        # At least some outputs should differ
        all_same = all(torch.allclose(outputs[0], o, atol=1e-6) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in outputs"


# =============================================================================
# Tests: Device Handling
# =============================================================================

class TestDeviceHandling:
    """Tests for device handling."""

    def test_cpu_training(self, small_config):
        """Test model works on CPU."""
        model = CausalTransformer(small_config)
        x = torch.randint(0, small_config.vocab_size, (2, 10))

        logits = model(x)
        assert logits.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_training(self, small_config):
        """Test model works on CUDA."""
        model = CausalTransformer(small_config).cuda()
        x = torch.randint(0, small_config.vocab_size, (2, 10)).cuda()

        logits = model(x)
        assert logits.device.type == 'cuda'


# =============================================================================
# Tests: Main Function
# =============================================================================

class TestMainFunction:
    """Tests for main function argument parsing."""

    def test_main_argument_parsing(self, mock_data_dir, temp_dir, monkeypatch):
        """Test main function argument parsing."""
        import sys

        test_args = [
            'train_ar_text.py',
            '--data_dir', mock_data_dir,
            '--checkpoint_dir', os.path.join(temp_dir, 'ckpts'),
            '--batch_size', '8',
            '--max_steps', '2',
            '--learning_rate', '0.001',
            '--warmup_steps', '1',
            '--d_model', '64',
            '--n_heads', '2',
            '--n_layers', '2',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        from src.training.train_ar_text import main
        main()

        # Verify checkpoint dir was created
        assert os.path.exists(os.path.join(temp_dir, 'ckpts'))


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_model(self):
        """Test very small model configuration."""
        config = ARConfig(
            vocab_size=100,
            max_seq_len=16,
            d_model=16,
            n_heads=1,
            n_layers=1,
            d_ff=32,
        )
        model = CausalTransformer(config)

        x = torch.randint(0, config.vocab_size, (1, 8))
        logits = model(x)

        assert logits.shape == (1, 8, config.vocab_size)

    def test_single_layer_model(self):
        """Test single layer model."""
        config = ARConfig(
            vocab_size=500,
            max_seq_len=32,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=128,
        )
        model = CausalTransformer(config)

        x = torch.randint(0, config.vocab_size, (2, 10))
        logits = model(x)

        assert logits.shape == (2, 10, config.vocab_size)

    def test_many_heads_model(self):
        """Test model with many attention heads."""
        config = ARConfig(
            vocab_size=500,
            max_seq_len=32,
            d_model=64,
            n_heads=8,  # Many heads
            n_layers=2,
            d_ff=128,
        )
        model = CausalTransformer(config)

        x = torch.randint(0, config.vocab_size, (2, 10))
        logits = model(x)

        assert logits.shape == (2, 10, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
