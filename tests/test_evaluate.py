#!/usr/bin/env python3
"""Tests for evaluate.py"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from src.evaluation.evaluate import (
    get_ngrams,
    compute_bleu,
    compute_self_bleu,
    compute_perplexity,
    load_model,
    generate_samples,
    evaluate_generation_quality,
)


class TestNgrams:
    """Tests for n-gram extraction."""

    def test_unigrams(self):
        tokens = [1, 2, 3, 4, 5]
        ngrams = get_ngrams(tokens, 1)
        assert ngrams == [(1,), (2,), (3,), (4,), (5,)]

    def test_bigrams(self):
        tokens = [1, 2, 3, 4]
        ngrams = get_ngrams(tokens, 2)
        assert ngrams == [(1, 2), (2, 3), (3, 4)]

    def test_trigrams(self):
        tokens = [1, 2, 3, 4]
        ngrams = get_ngrams(tokens, 3)
        assert ngrams == [(1, 2, 3), (2, 3, 4)]

    def test_empty_for_short_sequence(self):
        tokens = [1, 2]
        ngrams = get_ngrams(tokens, 3)
        assert ngrams == []

    def test_single_ngram(self):
        tokens = [1, 2, 3]
        ngrams = get_ngrams(tokens, 3)
        assert ngrams == [(1, 2, 3)]


class TestBLEU:
    """Tests for BLEU score computation."""

    def test_perfect_match(self):
        candidate = [1, 2, 3, 4, 5]
        references = [[1, 2, 3, 4, 5]]
        bleu = compute_bleu(candidate, references, max_n=4)
        assert bleu == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        candidate = [1, 2, 3, 4, 5]
        references = [[6, 7, 8, 9, 10]]
        bleu = compute_bleu(candidate, references, max_n=4)
        assert bleu == 0.0

    def test_partial_overlap(self):
        # Use longer sequences and lower max_n for partial overlap test
        candidate = [1, 2, 3, 4, 5, 6, 7, 8]
        references = [[1, 2, 3, 4, 9, 10, 11, 12]]
        bleu = compute_bleu(candidate, references, max_n=2)
        assert 0 < bleu < 1

    def test_multiple_references(self):
        candidate = [1, 2, 3, 4, 5]
        references = [[1, 2, 3, 6, 7], [1, 2, 3, 4, 8]]
        bleu = compute_bleu(candidate, references, max_n=4)
        assert 0 < bleu < 1

    def test_brevity_penalty(self):
        # Short candidate should be penalized
        candidate = [1, 2]
        references = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        bleu = compute_bleu(candidate, references, max_n=2)
        # Should be less than precision alone due to brevity penalty
        assert bleu < 1.0

    def test_unigram_only(self):
        candidate = [1, 2, 3]
        references = [[1, 2, 3]]
        bleu = compute_bleu(candidate, references, max_n=1)
        assert bleu == pytest.approx(1.0, abs=0.01)

    def test_short_candidate_no_ngrams(self):
        """Test BLEU with candidate too short for high-order n-grams."""
        candidate = [1, 2]  # Too short for 4-grams
        references = [[1, 2, 3, 4, 5, 6, 7, 8]]
        bleu = compute_bleu(candidate, references, max_n=4)
        # Should be 0 because no 4-grams possible
        assert bleu == 0.0

    def test_very_short_candidate(self):
        """Test BLEU with single token candidate."""
        candidate = [1]
        references = [[1, 2, 3, 4]]
        bleu = compute_bleu(candidate, references, max_n=2)
        # No bigrams possible, should return 0
        assert bleu == 0.0


class TestSelfBLEU:
    """Tests for Self-BLEU (diversity metric)."""

    def test_identical_samples_high_self_bleu(self):
        samples = [[1, 2, 3, 4, 5]] * 5
        self_bleu = compute_self_bleu(samples, max_n=4)
        assert self_bleu == pytest.approx(1.0, abs=0.01)

    def test_diverse_samples_low_self_bleu(self):
        samples = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
        self_bleu = compute_self_bleu(samples, max_n=4)
        assert self_bleu == 0.0

    def test_single_sample_returns_zero(self):
        samples = [[1, 2, 3, 4, 5]]
        self_bleu = compute_self_bleu(samples)
        assert self_bleu == 0.0

    def test_partially_similar(self):
        # Use longer sequences for meaningful n-gram overlap
        samples = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 11, 12, 13, 14, 15],
            [1, 2, 3, 4, 16, 17, 18, 19, 20, 21],
        ]
        self_bleu = compute_self_bleu(samples, max_n=2)
        assert 0 < self_bleu < 1


class TestLoadModel:
    """Tests for model loading."""

    def test_load_model_with_string_config(self, tmp_path):
        """Test loading checkpoint with string model config."""
        from src.core.model import create_model

        # Create and save a model with matching vocab_size/max_seq_len
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "test.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load it back
        loaded = load_model(str(checkpoint_path))
        assert loaded is not None

    def test_load_model_with_config_object(self, tmp_path):
        """Test loading checkpoint with ModelConfig object."""
        from src.core.model import create_model, ModelConfig, DiffusionTransformer

        # Create model and save with config object
        config = ModelConfig(
            d_model=256, n_heads=4, n_layers=4, d_ff=512,
            vocab_size=8192, max_seq_len=256
        )
        model = DiffusionTransformer(config)
        checkpoint = {
            "model_config": config,
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "test.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load it back
        loaded = load_model(str(checkpoint_path))
        assert loaded is not None


class TestPerplexity:
    """Tests for perplexity computation."""

    def test_perplexity_computation(self, tmp_path):
        """Test basic perplexity computation."""
        from src.core.model import create_model

        # Create tiny model with standard vocab
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        model.eval()

        # Create fake validation data
        val_data = torch.randint(1, 8192, (100, 256))
        val_path = tmp_path / "val.pt"
        torch.save(val_data, val_path)

        # Compute perplexity
        results = compute_perplexity(
            model, str(val_path),
            num_batches=2, batch_size=8
        )

        assert "avg_loss" in results
        assert "perplexity" in results
        assert results["perplexity"] > 0
        assert results["perplexity"] == pytest.approx(
            math.exp(results["avg_loss"]), rel=0.01
        )


class TestGenerateSamples:
    """Tests for sample generation."""

    def test_generate_samples(self):
        """Test that sample generation produces correct output format."""
        from src.core.model import create_model
        from src.core.diffusion import DiscreteDiffusion

        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        model.eval()
        diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

        samples = generate_samples(
            model, diffusion,
            num_samples=3,
            seq_len=32,
            steps=5,
            temperature=1.0,
            device="cpu"
        )

        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, list)
            assert all(isinstance(t, int) for t in sample)


class TestMainFunction:
    """Tests for main function."""

    def test_main_perplexity_only(self, tmp_path, monkeypatch):
        """Test main with --perplexity_only flag."""
        import sys
        from src.core.model import create_model

        # Create model checkpoint
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create validation data
        val_data = torch.randint(1, 8192, (50, 256))
        val_path = tmp_path / "val.pt"
        torch.save(val_data, val_path)

        # Mock command line args
        monkeypatch.setattr(sys, 'argv', [
            'evaluate.py',
            '--checkpoint', str(checkpoint_path),
            '--val_data', str(val_path),
            '--perplexity_only'
        ])

        # Import and run main
        from src.evaluation.evaluate import main
        main()  # Should complete without error


class TestIntegration:
    """Integration tests for evaluation."""

    def test_full_evaluation_flow(self, tmp_path):
        """Test complete evaluation pipeline with tiny model."""
        from src.core.model import create_model

        # Setup with standard vocab
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        model.eval()

        # Create fake validation data
        val_data = torch.randint(1, 8192, (50, 256))
        # Add BOS/EOS tokens
        val_data[:, 0] = 1  # BOS
        val_data[:, -1] = 2  # EOS
        val_path = tmp_path / "val.pt"
        torch.save(val_data, val_path)

        # Test perplexity
        ppl_results = compute_perplexity(
            model, str(val_path),
            num_batches=2, batch_size=8
        )
        assert ppl_results["perplexity"] > 0

        # Test generation quality (with very few samples for speed)
        gen_results = evaluate_generation_quality(
            model, str(val_path),
            num_samples=5,
            steps=5,
            temperature=1.0
        )
        assert "self_bleu" in gen_results
        assert "ref_bleu" in gen_results
        assert gen_results["self_bleu"] >= 0
        assert gen_results["ref_bleu"] >= 0


class TestMainFull:
    """Test main function with full evaluation."""

    def test_main_full_evaluation(self, tmp_path, monkeypatch, capsys):
        """Test main function with both perplexity and generation quality."""
        import sys
        from src.core.model import create_model

        # Create model checkpoint
        model = create_model("tiny", vocab_size=8192, max_seq_len=256)
        checkpoint = {
            "model_config": "tiny",
            "model_state_dict": model.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create validation data
        val_data = torch.randint(1, 8192, (50, 256))
        val_data[:, 0] = 1  # BOS
        val_data[:, -1] = 2  # EOS
        val_path = tmp_path / "val.pt"
        torch.save(val_data, val_path)

        # Mock command line args (no --perplexity_only)
        monkeypatch.setattr(sys, 'argv', [
            'evaluate.py',
            '--checkpoint', str(checkpoint_path),
            '--val_data', str(val_path),
            '--num_samples', '3',  # Small for speed
            '--steps', '3',  # Small for speed
        ])

        from src.evaluation.evaluate import main
        main()

        captured = capsys.readouterr()
        # Check that full output was generated
        assert "Perplexity Evaluation" in captured.out
        assert "Generation Quality Evaluation" in captured.out
        assert "Self-BLEU" in captured.out
        assert "Ref-BLEU" in captured.out
        assert "Summary" in captured.out
