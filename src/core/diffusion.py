#!/usr/bin/env python3
"""
Discrete diffusion process for language modeling (MDLM-style).

The diffusion process works by:
1. Forward process: Gradually mask tokens (add noise)
2. Reverse process: Iteratively predict and unmask (denoise)

Key insight: Unlike continuous diffusion (images), we use discrete masking.
At noise level t:
- t=0: Clean sequence (no masks)
- t=1: Fully masked sequence (all MASK tokens)

Usage:
    from src.core.diffusion import DiscreteDiffusion

    diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

    # Training
    loss, metrics = diffusion.training_losses(model, clean_tokens)

    # Sampling
    samples = diffusion.sample(model, batch_size=4, seq_len=128, num_steps=50)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class DiscreteDiffusion:
    """
    Discrete diffusion process using masking for language modeling.

    This implements a simplified version of MDLM (Masked Diffusion Language Model):
    - Forward process: Randomly mask tokens with probability dependent on t
    - Reverse process: Iteratively unmask tokens using model predictions
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        mask_token_id: int = 3,
        pad_token_id: int = 0,
        schedule: str = "cosine",
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            mask_token_id: ID of the MASK token
            pad_token_id: ID of the PAD token (never masked)
            schedule: Noise schedule - "cosine" or "linear"
        """
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.schedule = schedule

    def get_mask_rate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the mask rate for a given noise level t.

        Args:
            t: Noise level [batch_size] in range [0, 1]
               t=0 means clean (no masking)
               t=1 means fully masked

        Returns:
            Mask probability [batch_size] in range [0, 1]
        """
        if self.schedule == "linear":
            return t
        elif self.schedule == "cosine":
            # Cosine schedule: smoother near endpoints
            # At t=0: rate=0, at t=1: rate=1
            return 1 - torch.cos(t * torch.pi / 2)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def q_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: Add noise (masking) to clean tokens.

        Args:
            x: Clean token indices [batch_size, seq_len]
            t: Noise level [batch_size] in range [0, 1]

        Returns:
            x_noisy: Corrupted tokens [batch_size, seq_len]
            mask: Boolean mask of which positions were masked [batch_size, seq_len]
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Get mask probability for each sample in batch
        mask_rate = self.get_mask_rate(t)  # [batch_size]
        mask_rate = mask_rate.unsqueeze(1)  # [batch_size, 1]

        # Sample which positions to mask
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < mask_rate  # True where we mask

        # Don't mask padding tokens
        is_pad = (x == self.pad_token_id)
        mask = mask & ~is_pad

        # Apply masking
        x_noisy = x.clone()
        x_noisy[mask] = self.mask_token_id

        return x_noisy, mask

    def training_losses(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss for a batch of clean sequences.

        The loss is cross-entropy on predicting the original tokens
        at masked positions only.

        Args:
            model: DiffusionTransformer or ConditionalDiffusionLM
            x: Clean token indices [batch_size, seq_len]
            t: Optional noise levels [batch_size]. Sampled uniformly if not provided.
            attention_mask: Optional mask [batch_size, seq_len] (1=valid, 0=padding)
            encoder_output: Optional encoder output for cross-attention conditioning
                           [batch_size, encoder_seq_len, d_model]
            encoder_attention_mask: Optional mask for encoder [batch_size, encoder_seq_len]

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Sample random noise levels if not provided
        if t is None:
            t = torch.rand(batch_size, device=device)

        # Forward process: corrupt tokens
        x_noisy, mask = self.q_sample(x, t)

        # Get model predictions (with optional conditioning)
        logits = model(
            x_noisy, t,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Compute cross-entropy loss only on masked positions
        # Reshape for cross_entropy: [batch * seq_len, vocab_size]
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = x.view(-1)
        mask_flat = mask.view(-1).float()

        # Cross-entropy for all positions
        loss_per_token = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none'
        )

        # Average loss over masked positions only
        num_masked = mask_flat.sum().clamp(min=1)
        loss = (loss_per_token * mask_flat).sum() / num_masked

        # Compute metrics
        with torch.no_grad():
            # Accuracy on masked positions
            preds = logits.argmax(dim=-1)  # [batch, seq_len]
            correct = (preds == x) & mask
            accuracy = correct.float().sum() / num_masked

            # Mean mask rate
            mean_mask_rate = mask.float().mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "mask_rate": mean_mask_rate.item(),
            "num_masked": num_masked.item(),
        }

        return loss, metrics

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single reverse process step: denoise from t to t_next.

        Args:
            model: DiffusionTransformer
            x: Current noisy tokens [batch_size, seq_len]
            t: Current noise level [batch_size]
            t_next: Next noise level [batch_size] (t_next < t)
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            attention_mask: Optional attention mask
            encoder_output: Optional encoder output for conditioning
            encoder_attention_mask: Optional mask for encoder

        Returns:
            x_denoised: Less noisy tokens [batch_size, seq_len]
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Get model predictions (with optional conditioning)
        logits = model(
            x, t,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            # Zero out logits below top-k
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_vals[..., -1:]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float('-inf')),
                logits
            )

        # Sample from predicted distribution
        probs = F.softmax(logits, dim=-1)
        # Reshape for multinomial: [batch * seq_len, vocab_size]
        probs_flat = probs.view(-1, self.vocab_size)
        sampled_flat = torch.multinomial(probs_flat, num_samples=1)
        sampled = sampled_flat.view(batch_size, seq_len)

        # Determine which masks to keep vs. unmask
        # Current and next mask rates
        current_mask_rate = self.get_mask_rate(t).unsqueeze(1)  # [batch, 1]
        next_mask_rate = self.get_mask_rate(t_next).unsqueeze(1)  # [batch, 1]

        # Probability of keeping a mask (ratio of mask rates)
        # At t, we have current_mask_rate fraction masked
        # At t_next, we want next_mask_rate fraction masked
        # So we keep (next_mask_rate / current_mask_rate) of current masks
        keep_mask_prob = (next_mask_rate / current_mask_rate.clamp(min=1e-8)).clamp(max=1.0)

        # For each currently masked position, decide whether to keep it masked
        is_masked = (x == self.mask_token_id)
        rand = torch.rand(batch_size, seq_len, device=device)
        keep_mask = rand < keep_mask_prob

        # Unmask positions: currently masked AND not keeping the mask
        unmask = is_masked & ~keep_mask

        # Update tokens
        x_denoised = x.clone()
        x_denoised[unmask] = sampled[unmask]

        return x_denoised

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = "cuda",
        prompt: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate sequences by iterative denoising.

        Args:
            model: DiffusionTransformer
            batch_size: Number of sequences to generate
            seq_len: Length of sequences to generate
            num_steps: Number of denoising steps
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Optional top-k filtering
            device: Device to use
            prompt: Optional prompt tokens [batch_size, prompt_len] (for prefix conditioning)
            attention_mask: Optional attention mask
            encoder_output: Optional encoder output for cross-attention conditioning
            encoder_attention_mask: Optional mask for encoder

        Returns:
            samples: Generated token indices [batch_size, seq_len]
        """
        model.eval()

        # Move encoder output to device if provided
        if encoder_output is not None:
            encoder_output = encoder_output.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)

        # Handle prompt (for prefix-style conditioning)
        if prompt is not None:
            prompt_len = prompt.shape[1]
            assert prompt.shape[0] == batch_size, "Prompt batch size must match"
            assert prompt_len < seq_len, "Prompt must be shorter than seq_len"

            # Start with prompt + masks
            x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
            x[:, :prompt_len] = prompt
        else:
            prompt_len = 0
            # Start with all masks
            x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)

        # Create timestep schedule: t goes from 1.0 to 0.0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        # Iterative denoising
        for i in range(num_steps):
            t = timesteps[i].expand(batch_size)
            t_next = timesteps[i + 1].expand(batch_size)

            x = self.p_sample_step(
                model, x, t, t_next,
                temperature=temperature,
                top_k=top_k,
                attention_mask=attention_mask,
                encoder_output=encoder_output,
                encoder_attention_mask=encoder_attention_mask,
            )

            # Keep prompt fixed (for prefix conditioning)
            if prompt is not None:
                x[:, :prompt_len] = prompt

        return x

    def sample_with_trajectory(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, list]:
        """
        Sample with trajectory recording (for visualization).

        Returns:
            samples: Final generated tokens [batch_size, seq_len]
            trajectory: List of intermediate states
        """
        model.eval()
        trajectory = []

        x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
        trajectory.append(x.clone())

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i].expand(batch_size)
            t_next = timesteps[i + 1].expand(batch_size)

            x = self.p_sample_step(model, x, t, t_next, temperature, top_k)
            trajectory.append(x.clone())

        return x, trajectory


if __name__ == "__main__":
    from src.core.model import create_model

    print("=" * 60)
    print("Diffusion Process Tests")
    print("=" * 60)

    # Create model and diffusion process
    model = create_model("tiny")
    diffusion = DiscreteDiffusion(vocab_size=8192, mask_token_id=3)

    # Test forward process
    print("\n1. Forward Process (q_sample):")
    batch_size, seq_len = 4, 32
    x = torch.randint(4, 8192, (batch_size, seq_len))  # Avoid special tokens

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.full((batch_size,), t_val)
        x_noisy, mask = diffusion.q_sample(x, t)
        mask_frac = mask.float().mean().item()
        print(f"  t={t_val:.2f}: mask_rate={mask_frac:.3f}")

    # Test training loss
    print("\n2. Training Loss:")
    x = torch.randint(4, 8192, (batch_size, seq_len))
    loss, metrics = diffusion.training_losses(model, x)
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mask rate: {metrics['mask_rate']:.4f}")

    # Test sampling
    print("\n3. Sampling:")
    samples = diffusion.sample(
        model,
        batch_size=2,
        seq_len=16,
        num_steps=10,
        device="cpu"
    )
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample[0]: {samples[0].tolist()}")

    # Test sampling with prompt
    print("\n4. Sampling with Prompt:")
    prompt = torch.randint(4, 100, (2, 4))  # Short prompt
    samples = diffusion.sample(
        model,
        batch_size=2,
        seq_len=16,
        num_steps=10,
        device="cpu",
        prompt=prompt
    )
    print(f"  Prompt: {prompt[0].tolist()}")
    print(f"  Sample: {samples[0].tolist()}")
    # Verify prompt is preserved
    assert torch.equal(samples[:, :4], prompt), "Prompt should be preserved!"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
