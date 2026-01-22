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
        top_p: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        allow_recorruption: bool = False,
        recorruption_rate: float = 0.1,
    ) -> torch.Tensor:
        """
        Single reverse process step: denoise from t to t_next.

        Implements proper MDLM transition probabilities:
        - For masked positions: unmask with prob = 1 - (next_mask_rate / current_mask_rate)
        - For unmasked positions: optionally re-corrupt with small probability

        Args:
            model: DiffusionTransformer
            x: Current noisy tokens [batch_size, seq_len]
            t: Current noise level [batch_size]
            t_next: Next noise level [batch_size] (t_next < t)
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            top_p: Optional nucleus (top-p) filtering
            attention_mask: Optional attention mask
            encoder_output: Optional encoder output for conditioning
            encoder_attention_mask: Optional mask for encoder
            allow_recorruption: Whether to allow re-masking unmasked tokens
            recorruption_rate: Fraction of step's unmask prob to use for re-corruption

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
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_vals[..., -1:]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float('-inf')),
                logits
            )

        # Apply nucleus (top-p) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            # Shift to include first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            # Scatter back to original indexing
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample from predicted distribution
        probs = F.softmax(logits, dim=-1)
        # Reshape for multinomial: [batch * seq_len, vocab_size]
        probs_flat = probs.view(-1, self.vocab_size)
        sampled_flat = torch.multinomial(probs_flat, num_samples=1)
        sampled = sampled_flat.view(batch_size, seq_len)

        # MDLM transition probabilities
        current_mask_rate = self.get_mask_rate(t).unsqueeze(1)  # [batch, 1]
        next_mask_rate = self.get_mask_rate(t_next).unsqueeze(1)  # [batch, 1]

        # Probability of unmasking = (current - next) / current
        # This is the proper MDLM formulation
        unmask_prob = ((current_mask_rate - next_mask_rate) /
                       current_mask_rate.clamp(min=1e-8)).clamp(min=0.0, max=1.0)

        # Identify masked and unmasked positions
        is_masked = (x == self.mask_token_id)
        is_pad = (x == self.pad_token_id)

        # Decide which masked positions to unmask
        rand = torch.rand(batch_size, seq_len, device=device)
        unmask = is_masked & (rand < unmask_prob)

        # Start with current tokens
        x_denoised = x.clone()

        # Unmask selected positions with sampled tokens
        x_denoised[unmask] = sampled[unmask]

        # Optional: re-corruption of unmasked positions (Bug 3 fix)
        # This allows some exploration during sampling
        if allow_recorruption and recorruption_rate > 0:
            # Re-mask a small fraction of currently unmasked positions
            # Probability scales with remaining noise level
            recorrupt_prob = recorruption_rate * next_mask_rate
            recorrupt = (~is_masked) & (~is_pad) & (rand < recorrupt_prob)
            x_denoised[recorrupt] = self.mask_token_id

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
        top_p: Optional[float] = None,
        device: str = "cuda",
        prompt: Optional[torch.Tensor] = None,
        prompt_mode: str = "fixed",
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        allow_recorruption: bool = False,
        recorruption_rate: float = 0.1,
        temperature_schedule: Optional[str] = None,
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
            top_p: Optional nucleus (top-p) filtering (e.g., 0.9, 0.95)
            device: Device to use
            prompt: Optional prompt tokens [batch_size, prompt_len] (for prefix conditioning)
            prompt_mode: How to handle prompt:
                - "fixed": Hard-fix prompt at end (original behavior)
                - "soft": Allow model to see prompt but generate freely (bidirectional)
            attention_mask: Optional attention mask
            encoder_output: Optional encoder output for cross-attention conditioning
            encoder_attention_mask: Optional mask for encoder
            allow_recorruption: Whether to allow re-masking during sampling
            recorruption_rate: Fraction of step's unmask prob for re-corruption
            temperature_schedule: Optional schedule for temperature:
                - None: Use constant temperature
                - "linear_decay": 1.5 -> 0.5 over steps
                - "cosine_decay": Cosine annealing from 1.5 to 0.5

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
            prompt = prompt.to(device)
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

            # Compute temperature for this step
            if temperature_schedule is None:
                step_temp = temperature
            elif temperature_schedule == "linear_decay":
                # Linear decay: 1.5 -> 0.5
                progress = i / max(num_steps - 1, 1)
                step_temp = 1.5 - 1.0 * progress
            elif temperature_schedule == "cosine_decay":
                # Cosine decay: 1.5 -> 0.5
                progress = i / max(num_steps - 1, 1)
                step_temp = 0.5 + 1.0 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
                step_temp = step_temp.item()
            else:
                step_temp = temperature

            x = self.p_sample_step(
                model, x, t, t_next,
                temperature=step_temp,
                top_k=top_k,
                top_p=top_p,
                attention_mask=attention_mask,
                encoder_output=encoder_output,
                encoder_attention_mask=encoder_attention_mask,
                allow_recorruption=allow_recorruption,
                recorruption_rate=recorruption_rate,
            )

            # Handle prompt based on mode
            if prompt is not None:
                if prompt_mode == "fixed":
                    # Original behavior: hard-fix prompt every step
                    x[:, :prompt_len] = prompt
                elif prompt_mode == "soft":
                    # Soft mode: only ensure prompt positions aren't masked
                    # (they participate in bidirectional attention normally)
                    is_prompt_masked = (x[:, :prompt_len] == self.mask_token_id)
                    x[:, :prompt_len] = torch.where(
                        is_prompt_masked,
                        prompt,
                        x[:, :prompt_len]
                    )

        return x

    def sample_with_trajectory(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = "cuda",
        temperature_schedule: Optional[str] = None,
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

            # Compute temperature for this step
            if temperature_schedule is None:
                step_temp = temperature
            elif temperature_schedule == "linear_decay":
                progress = i / max(num_steps - 1, 1)
                step_temp = 1.5 - 1.0 * progress
            elif temperature_schedule == "cosine_decay":
                progress = i / max(num_steps - 1, 1)
                step_temp = 0.5 + 1.0 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
                step_temp = step_temp.item()
            else:
                step_temp = temperature

            x = self.p_sample_step(
                model, x, t, t_next,
                temperature=step_temp,
                top_k=top_k,
                top_p=top_p,
            )
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
