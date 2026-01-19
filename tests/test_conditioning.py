#!/usr/bin/env python3
"""
Tests for conditional generation components.

Run with: pytest test_conditioning.py -v
"""

import pytest
import torch

from src.core.model import (
    ModelConfig,
    TextEncoder,
    DiffusionTransformer,
    ConditionalDiffusionLM,
    TransformerBlock,
    create_encoder,
    create_conditional_model,
    create_model,
    MODEL_CONFIGS,
)
from src.core.diffusion import DiscreteDiffusion
from src.data.data_prep import split_into_sentences, extract_conditional_pairs


# =============================================================================
# Tests: TextEncoder
# =============================================================================

class TestTextEncoder:
    """Tests for TextEncoder."""

    @pytest.fixture
    def encoder(self):
        config = ModelConfig(
            d_model=128, n_heads=4, n_layers=2, d_ff=256,
            vocab_size=256, max_seq_len=32, dropout=0.0,
        )
        return TextEncoder(config)

    def test_output_shape(self, encoder):
        """Test encoder output shape."""
        x = torch.randint(0, 256, (4, 32))
        out = encoder(x)
        assert out.shape == (4, 32, 128)

    def test_with_attention_mask(self, encoder):
        """Test encoder with attention mask."""
        x = torch.randint(0, 256, (4, 32))
        mask = torch.ones(4, 32)
        mask[:, 16:] = 0  # Mask second half
        out = encoder(x, attention_mask=mask)
        assert out.shape == (4, 32, 128)

    def test_parameter_count(self, encoder):
        """Test parameter counting."""
        count = encoder.count_parameters()
        assert count > 0
        assert count == encoder.count_parameters(trainable_only=True)

    def test_different_seq_lengths(self, encoder):
        """Test encoder with different sequence lengths."""
        for seq_len in [8, 16, 32]:
            x = torch.randint(0, 256, (2, seq_len))
            out = encoder(x)
            assert out.shape == (2, seq_len, 128)


# =============================================================================
# Tests: TransformerBlock with Cross-Attention
# =============================================================================

class TestCrossAttention:
    """Tests for TransformerBlock with cross-attention."""

    @pytest.fixture
    def block_with_cross_attn(self):
        return TransformerBlock(
            d_model=128, n_heads=4, d_ff=256, dropout=0.0,
            has_cross_attention=True,
        )

    @pytest.fixture
    def block_without_cross_attn(self):
        return TransformerBlock(
            d_model=128, n_heads=4, d_ff=256, dropout=0.0,
            has_cross_attention=False,
        )

    def test_cross_attention_flag(self, block_with_cross_attn, block_without_cross_attn):
        """Test has_cross_attention attribute."""
        assert block_with_cross_attn.has_cross_attention is True
        assert block_without_cross_attn.has_cross_attention is False

    def test_cross_attention_layers_exist(self, block_with_cross_attn):
        """Test that cross-attention layers are created."""
        assert hasattr(block_with_cross_attn, 'cross_attn')
        assert hasattr(block_with_cross_attn, 'norm_cross')

    def test_no_cross_attention_layers(self, block_without_cross_attn):
        """Test that cross-attention layers are not created when disabled."""
        assert not hasattr(block_without_cross_attn, 'cross_attn')
        assert not hasattr(block_without_cross_attn, 'norm_cross')

    def test_forward_without_encoder_output(self, block_with_cross_attn):
        """Test forward pass without encoder output."""
        x = torch.randn(4, 16, 128)
        out = block_with_cross_attn(x)
        assert out.shape == x.shape

    def test_forward_with_encoder_output(self, block_with_cross_attn):
        """Test forward pass with encoder output."""
        x = torch.randn(4, 16, 128)
        encoder_output = torch.randn(4, 32, 128)
        out = block_with_cross_attn(x, encoder_output=encoder_output)
        assert out.shape == x.shape

    def test_cross_attention_affects_output(self, block_with_cross_attn):
        """Test that cross-attention changes the output."""
        x = torch.randn(4, 16, 128)
        encoder_output = torch.randn(4, 32, 128)

        out_without = block_with_cross_attn(x)
        out_with = block_with_cross_attn(x, encoder_output=encoder_output)

        assert not torch.allclose(out_without, out_with)


# =============================================================================
# Tests: DiffusionTransformer with Cross-Attention
# =============================================================================

class TestDecoderWithCrossAttention:
    """Tests for DiffusionTransformer with cross-attention enabled."""

    @pytest.fixture
    def decoder(self):
        config = ModelConfig(
            d_model=128, n_heads=4, n_layers=2, d_ff=256,
            vocab_size=256, max_seq_len=64, dropout=0.0,
            has_cross_attention=True,
        )
        return DiffusionTransformer(config)

    def test_output_shape_without_encoder(self, decoder):
        """Test decoder output shape without encoder output."""
        x = torch.randint(0, 256, (4, 32))
        t = torch.rand(4)
        out = decoder(x, t)
        assert out.shape == (4, 32, 256)

    def test_output_shape_with_encoder(self, decoder):
        """Test decoder output shape with encoder output."""
        x = torch.randint(0, 256, (4, 32))
        t = torch.rand(4)
        encoder_output = torch.randn(4, 16, 128)
        out = decoder(x, t, encoder_output=encoder_output)
        assert out.shape == (4, 32, 256)

    def test_encoder_output_affects_predictions(self, decoder):
        """Test that encoder output affects predictions."""
        x = torch.randint(0, 256, (4, 32))
        t = torch.rand(4)
        encoder_output = torch.randn(4, 16, 128)

        with torch.no_grad():
            out_without = decoder(x, t)
            out_with = decoder(x, t, encoder_output=encoder_output)

        assert not torch.allclose(out_without, out_with)


# =============================================================================
# Tests: ConditionalDiffusionLM
# =============================================================================

class TestConditionalDiffusionLM:
    """Tests for the complete conditional model."""

    @pytest.fixture
    def model(self):
        return create_conditional_model(
            "tiny", "tiny", vocab_size=256,
            encoder_max_seq_len=32, decoder_max_seq_len=64,
        )

    def test_forward_pass(self, model):
        """Test forward pass of conditional model."""
        enc_input = torch.randint(0, 256, (4, 32))
        dec_input = torch.randint(0, 256, (4, 64))
        t = torch.rand(4)

        logits = model(dec_input, t, enc_input)
        assert logits.shape == (4, 64, 256)

    def test_with_attention_masks(self, model):
        """Test forward pass with attention masks."""
        enc_input = torch.randint(0, 256, (4, 32))
        dec_input = torch.randint(0, 256, (4, 64))
        t = torch.rand(4)

        enc_mask = torch.ones(4, 32)
        dec_mask = torch.ones(4, 64)

        logits = model(
            dec_input, t, enc_input,
            decoder_attention_mask=dec_mask,
            encoder_attention_mask=enc_mask,
        )
        assert logits.shape == (4, 64, 256)

    def test_freeze_decoder(self, model):
        """Test freezing decoder for stage 2 training."""
        # Before freezing - all trainable
        trainable_before = model.count_parameters(trainable_only=True)

        # Freeze decoder
        model.freeze_decoder()

        # After freezing - only encoder and cross-attention trainable
        trainable_after = model.count_parameters(trainable_only=True)

        assert trainable_after < trainable_before

        # Check encoder is still trainable
        for param in model.encoder.parameters():
            assert param.requires_grad is True

        # Check self-attention in decoder is frozen
        for block in model.decoder.blocks:
            for param in block.self_attn.parameters():
                assert param.requires_grad is False

        # Check cross-attention in decoder is trainable
        for block in model.decoder.blocks:
            if block.has_cross_attention:
                for param in block.cross_attn.parameters():
                    assert param.requires_grad is True

    def test_unfreeze_all(self, model):
        """Test unfreezing all parameters."""
        model.freeze_decoder()
        model.unfreeze_all()

        for param in model.parameters():
            assert param.requires_grad is True

    def test_invalid_config_no_cross_attention(self):
        """Test that decoder without cross-attention raises error."""
        encoder_config = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256)
        decoder_config = ModelConfig(
            d_model=128, n_heads=4, n_layers=2, d_ff=256,
            has_cross_attention=False,  # Should raise error
        )

        with pytest.raises(ValueError, match="has_cross_attention"):
            ConditionalDiffusionLM(encoder_config, decoder_config)

    def test_invalid_config_mismatched_d_model(self):
        """Test that mismatched d_model raises error."""
        encoder_config = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256)
        decoder_config = ModelConfig(
            d_model=256,  # Different from encoder
            n_heads=4, n_layers=2, d_ff=512,
            has_cross_attention=True,
        )

        with pytest.raises(ValueError, match="d_model"):
            ConditionalDiffusionLM(encoder_config, decoder_config)


# =============================================================================
# Tests: Conditional Training
# =============================================================================

class TestConditionalTraining:
    """Tests for conditional training with diffusion."""

    @pytest.fixture
    def model_and_diffusion(self):
        model = create_model(
            "tiny", vocab_size=256, max_seq_len=64,
            has_cross_attention=True,
        )
        diffusion = DiscreteDiffusion(vocab_size=256, mask_token_id=3)
        return model, diffusion

    def test_training_loss_with_encoder_output(self, model_and_diffusion):
        """Test training loss computation with encoder output."""
        model, diffusion = model_and_diffusion

        x = torch.randint(5, 256, (4, 32))  # Clean tokens
        encoder_output = torch.randn(4, 16, 256)

        loss, metrics = diffusion.training_losses(
            model, x, encoder_output=encoder_output
        )

        assert loss.item() > 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_sampling_with_encoder_output(self, model_and_diffusion):
        """Test sampling with encoder conditioning."""
        model, diffusion = model_and_diffusion

        encoder_output = torch.randn(2, 16, 256)

        samples = diffusion.sample(
            model, batch_size=2, seq_len=32, num_steps=5,
            device="cpu", encoder_output=encoder_output,
        )

        assert samples.shape == (2, 32)
        assert (samples != diffusion.mask_token_id).any()


# =============================================================================
# Tests: Sentence Splitting
# =============================================================================

class TestSentenceSplitting:
    """Tests for sentence splitting function."""

    def test_basic_splitting(self):
        """Test basic sentence splitting."""
        text = "Hello world. How are you? I am fine."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine."

    def test_abbreviations(self):
        """Test handling of abbreviations."""
        text = "Mr. Smith went to Dr. Jones. They talked."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "Mr. Smith" in sentences[0]
        assert "Dr. Jones" in sentences[0]

    def test_exclamation_and_question(self):
        """Test ! and ? as sentence endings."""
        text = "Wow! Is that true? Yes it is."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_single_sentence(self):
        """Test text with single sentence."""
        text = "Just one sentence here."
        sentences = split_into_sentences(text)
        assert len(sentences) == 1

    def test_empty_string(self):
        """Test empty string."""
        sentences = split_into_sentences("")
        assert len(sentences) == 0


# =============================================================================
# Tests: Model Config Sizes
# =============================================================================

class TestModelConfigs:
    """Tests for new model configurations."""

    @pytest.mark.parametrize("config_name", ["xlarge", "xxlarge"])
    def test_new_configs_exist(self, config_name):
        """Test that xlarge and xxlarge configs exist."""
        assert config_name in MODEL_CONFIGS

    def test_xlarge_config_values(self):
        """Test xlarge config has expected values."""
        config = MODEL_CONFIGS["xlarge"]
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12

    def test_xxlarge_config_values(self):
        """Test xxlarge config has expected values."""
        config = MODEL_CONFIGS["xxlarge"]
        assert config.d_model == 1024
        assert config.n_heads == 16
        assert config.n_layers == 16

    def test_xlarge_model_creation(self):
        """Test creating xlarge model."""
        model = create_model("xlarge")
        assert model.count_parameters() > MODEL_CONFIGS["large"].d_model * 1000

    def test_xxlarge_larger_than_xlarge(self):
        """Test xxlarge is larger than xlarge."""
        xlarge = create_model("xlarge")
        xxlarge = create_model("xxlarge")
        assert xxlarge.count_parameters() > xlarge.count_parameters()


# =============================================================================
# Tests: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_encoder(self):
        """Test create_encoder function."""
        encoder = create_encoder("tiny", vocab_size=256, max_seq_len=32)
        assert isinstance(encoder, TextEncoder)
        assert encoder.config.vocab_size == 256

    def test_create_conditional_model(self):
        """Test create_conditional_model function."""
        model = create_conditional_model(
            "tiny", "small", vocab_size=256,
            encoder_max_seq_len=32, decoder_max_seq_len=64,
        )
        assert isinstance(model, ConditionalDiffusionLM)

    def test_create_model_with_cross_attention(self):
        """Test create_model with has_cross_attention kwarg."""
        model = create_model("tiny", has_cross_attention=True)
        assert model.config.has_cross_attention is True
        for block in model.blocks:
            assert block.has_cross_attention is True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
