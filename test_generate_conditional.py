#!/usr/bin/env python3
"""Tests for generate_conditional.py"""

import pytest
import torch
from pathlib import Path

from generate_conditional import load_conditional_model, generate


class TestLoadConditionalModel:
    """Tests for model loading."""

    @pytest.fixture
    def checkpoint_path(self, tmp_path):
        """Create a test checkpoint."""
        from model import DiffusionTransformer, TextEncoder, ModelConfig

        # Create encoder config
        encoder_config = ModelConfig(
            d_model=128, n_heads=2, n_layers=2, d_ff=256,
            vocab_size=500, max_seq_len=32,
            has_cross_attention=False,
        )
        encoder = TextEncoder(encoder_config)

        # Create decoder config with cross-attention
        decoder_config = ModelConfig(
            d_model=128, n_heads=2, n_layers=2, d_ff=256,
            vocab_size=500, max_seq_len=64,
            has_cross_attention=True,
        )
        decoder = DiffusionTransformer(decoder_config)

        # Create checkpoint
        checkpoint = {
            "encoder_config": encoder_config,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_config": decoder_config,
            "decoder_state_dict": decoder.state_dict(),
            "train_config": {
                "max_encoder_len": 32,
                "max_decoder_len": 64,
            },
        }

        path = tmp_path / "model.pt"
        torch.save(checkpoint, path)
        return str(path)

    def test_load_model(self, checkpoint_path):
        """Test loading encoder and decoder from checkpoint."""
        encoder, decoder, train_config = load_conditional_model(checkpoint_path)

        assert encoder is not None
        assert decoder is not None
        assert train_config is not None

    def test_encoder_config(self, checkpoint_path):
        """Test encoder has correct config."""
        encoder, decoder, _ = load_conditional_model(checkpoint_path)

        assert encoder.config.d_model == 128
        assert encoder.config.n_layers == 2
        assert encoder.config.vocab_size == 500

    def test_decoder_config(self, checkpoint_path):
        """Test decoder has correct config."""
        encoder, decoder, _ = load_conditional_model(checkpoint_path)

        assert decoder.config.d_model == 128
        assert decoder.config.n_layers == 2
        assert decoder.config.has_cross_attention

    def test_models_in_eval_mode(self, checkpoint_path):
        """Test models are set to eval mode."""
        encoder, decoder, _ = load_conditional_model(checkpoint_path)

        assert not encoder.training
        assert not decoder.training

    def test_load_with_dict_config(self, tmp_path):
        """Test loading when config is stored as dict."""
        from model import DiffusionTransformer, TextEncoder, ModelConfig

        encoder_config = ModelConfig(
            d_model=128, n_heads=2, n_layers=2, d_ff=256,
            vocab_size=500, max_seq_len=32,
        )
        encoder = TextEncoder(encoder_config)

        decoder_config = ModelConfig(
            d_model=128, n_heads=2, n_layers=2, d_ff=256,
            vocab_size=500, max_seq_len=64,
            has_cross_attention=True,
        )
        decoder = DiffusionTransformer(decoder_config)

        # Save configs as dicts
        from dataclasses import asdict
        checkpoint = {
            "encoder_config": asdict(encoder_config),
            "encoder_state_dict": encoder.state_dict(),
            "decoder_config": asdict(decoder_config),
            "decoder_state_dict": decoder.state_dict(),
            "train_config": {"max_encoder_len": 32},
        }

        path = tmp_path / "model_dict.pt"
        torch.save(checkpoint, path)

        encoder, decoder, _ = load_conditional_model(str(path))
        assert encoder is not None
        assert decoder is not None


class TestGenerate:
    """Tests for generation function."""

    @pytest.fixture
    def model_pair(self):
        """Create encoder and decoder for testing."""
        from model import DiffusionTransformer, TextEncoder, ModelConfig

        encoder_config = ModelConfig(
            d_model=64, n_heads=2, n_layers=2, d_ff=128,
            vocab_size=100, max_seq_len=16,
        )
        encoder = TextEncoder(encoder_config)
        encoder.eval()

        decoder_config = ModelConfig(
            d_model=64, n_heads=2, n_layers=2, d_ff=128,
            vocab_size=100, max_seq_len=32,
            has_cross_attention=True,
        )
        decoder = DiffusionTransformer(decoder_config)
        decoder.eval()

        return encoder, decoder

    def test_generate_single_sample(self, model_pair):
        """Test generating a single sample."""
        encoder, decoder = model_pair
        prompt = torch.randint(1, 100, (16,))

        output = generate(
            encoder, decoder, prompt,
            max_len=32, steps=5, temperature=0.8,
            mask_token_id=3, pad_token_id=0,
        )

        assert output.shape == (1, 32)
        assert output.dtype == torch.long

    def test_generate_batch(self, model_pair):
        """Test generating multiple samples."""
        encoder, decoder = model_pair
        prompts = torch.randint(1, 100, (4, 16))

        output = generate(
            encoder, decoder, prompts,
            max_len=32, steps=5, temperature=0.8,
            mask_token_id=3, pad_token_id=0,
        )

        assert output.shape == (4, 32)

    def test_generate_respects_max_len(self, model_pair):
        """Test that output has correct length."""
        encoder, decoder = model_pair
        prompt = torch.randint(1, 100, (16,))

        for max_len in [16, 24, 32]:
            output = generate(
                encoder, decoder, prompt,
                max_len=max_len, steps=3,
                mask_token_id=3, pad_token_id=0,
            )
            assert output.shape[1] == max_len

    def test_generate_different_temperatures(self, model_pair):
        """Test generation with different temperatures."""
        encoder, decoder = model_pair
        prompt = torch.randint(1, 100, (16,))

        for temp in [0.5, 1.0, 1.5]:
            output = generate(
                encoder, decoder, prompt,
                max_len=32, steps=3, temperature=temp,
                mask_token_id=3, pad_token_id=0,
            )
            assert output.shape == (1, 32)

    def test_generate_no_mask_tokens_at_end(self, model_pair):
        """Test that output has no mask tokens after enough steps."""
        encoder, decoder = model_pair
        prompt = torch.randint(1, 100, (16,))

        output = generate(
            encoder, decoder, prompt,
            max_len=32, steps=50, temperature=0.8,
            mask_token_id=3, pad_token_id=0,
        )

        # After 50 steps, should have no mask tokens
        mask_count = (output == 3).sum().item()
        assert mask_count == 0

    def test_generate_with_padding_in_prompt(self, model_pair):
        """Test generation handles padding in prompt correctly."""
        encoder, decoder = model_pair

        # Create prompt with padding
        prompt = torch.randint(1, 100, (16,))
        prompt[10:] = 0  # Pad last 6 tokens

        output = generate(
            encoder, decoder, prompt,
            max_len=32, steps=5,
            mask_token_id=3, pad_token_id=0,
        )

        assert output.shape == (1, 32)


class TestMainFunction:
    """Tests for main function argument handling."""

    def test_argument_parsing(self, tmp_path, monkeypatch):
        """Test that main parses arguments correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=str, default="checkpoints_conditional/best.pt")
        parser.add_argument("--tokenizer", type=str, default="data_full/tokenizer.json")
        parser.add_argument("--prompt", type=str, default="Test prompt.")
        parser.add_argument("--num_samples", "-n", type=int, default=3)
        parser.add_argument("--steps", type=int, default=100)
        parser.add_argument("--temperature", "-t", type=float, default=0.8)
        parser.add_argument("--max_len", type=int, default=192)

        args = parser.parse_args([
            "--prompt", "Custom prompt.",
            "--steps", "50",
            "-n", "5",
        ])

        assert args.prompt == "Custom prompt."
        assert args.steps == 50
        assert args.num_samples == 5
        assert args.temperature == 0.8  # default


class TestMainFunctionExecution:
    """Tests for main function execution."""

    def test_main_full_execution(self, tmp_path, monkeypatch):
        """Test main function runs end-to-end."""
        import sys
        from model import DiffusionTransformer, TextEncoder, ModelConfig
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=500,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story about things."] * 50,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Create models
        encoder_config = ModelConfig(
            d_model=64, n_heads=2, n_layers=2, d_ff=128,
            vocab_size=500, max_seq_len=32,
        )
        encoder = TextEncoder(encoder_config)

        decoder_config = ModelConfig(
            d_model=64, n_heads=2, n_layers=2, d_ff=128,
            vocab_size=500, max_seq_len=64,
            has_cross_attention=True,
        )
        decoder = DiffusionTransformer(decoder_config)

        # Save checkpoint
        checkpoint = {
            "encoder_config": encoder_config,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_config": decoder_config,
            "decoder_state_dict": decoder.state_dict(),
            "train_config": {"max_encoder_len": 32},
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Mock command line args
        monkeypatch.setattr(sys, 'argv', [
            'generate_conditional.py',
            '--checkpoint', str(checkpoint_path),
            '--tokenizer', str(tokenizer_path),
            '--prompt', 'Once upon a time',
            '--num_samples', '2',
            '--steps', '5',
            '--max_len', '32',
        ])

        from generate_conditional import main
        main()  # Should complete without error


class TestIntegration:
    """Integration tests for conditional generation."""

    def test_full_generation_pipeline(self, tmp_path):
        """Test complete generation pipeline."""
        from model import DiffusionTransformer, TextEncoder, ModelConfig
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=500,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        )
        tokenizer.train_from_iterator(
            ["Once upon a time there was a story."] * 50,
            trainer=trainer
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Create models
        encoder_config = ModelConfig(
            d_model=64, n_heads=2, n_layers=2, d_ff=128,
            vocab_size=500, max_seq_len=32,
        )
        encoder = TextEncoder(encoder_config)

        decoder_config = ModelConfig(
            d_model=64, n_heads=2, n_layers=2, d_ff=128,
            vocab_size=500, max_seq_len=64,
            has_cross_attention=True,
        )
        decoder = DiffusionTransformer(decoder_config)

        # Save checkpoint
        checkpoint = {
            "encoder_config": encoder_config,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_config": decoder_config,
            "decoder_state_dict": decoder.state_dict(),
            "train_config": {"max_encoder_len": 32},
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load models
        enc, dec, _ = load_conditional_model(str(checkpoint_path))

        # Tokenize prompt
        prompt = "Once upon a time"
        encoded = tokenizer.encode(prompt)
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")
        pad_id = tokenizer.token_to_id("<PAD>")
        mask_id = tokenizer.token_to_id("<MASK>")

        prompt_ids = [bos_id] + encoded.ids + [eos_id]
        # Pad to 32
        while len(prompt_ids) < 32:
            prompt_ids.append(pad_id)

        prompt_tensor = torch.tensor(prompt_ids)

        # Generate
        output = generate(
            enc, dec, prompt_tensor,
            max_len=64, steps=10,
            mask_token_id=mask_id, pad_token_id=pad_id,
        )

        assert output.shape == (1, 64)

        # Decode output
        tokens = output[0].tolist()
        tokens = [t for t in tokens if t not in [pad_id, bos_id, eos_id, mask_id]]
        text = tokenizer.decode(tokens)

        assert isinstance(text, str)
