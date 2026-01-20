#!/usr/bin/env python3
"""
Sparse Distribution Diffusion (SDD) for Text Generation.

This module implements a novel diffusion approach that operates on sparse
probability distributions over token embeddings rather than discrete tokens.

Key Idea:
---------
Instead of representing each position as a single token (discrete) or a
full distribution over V tokens (dense), we use k weighted embedding vectors:

    position_i = Σ_{j=1}^{k} prob_ij × embedding_j

This is mathematically a sum of k Dirac delta functions in continuous
embedding space - hence "Sparse Distribution Diffusion".

Why this matters:
- vs Discrete (D3PM, MDLM): We maintain probability distributions, not hard tokens
- vs Full Embedding (Diffusion-LM): We're sparse (k terms), not dense (V terms)
- Novel middle ground: Expressive but tractable

Key concepts:
- State: (probs, embeds, indices) representing top-k tokens per position
- Noise: Mix toward uniform + Gaussian noise on embeddings
- Denoising: Refine distributions iteratively
- IGNORE token: Enables variable-length generation
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SDDConfig:
    """Configuration for Sparse Distribution Diffusion."""
    vocab_size: int = 8192
    embed_dim: int = 64
    k: int = 8  # Number of top-k tokens per position

    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 3  # Not used in SDD, but kept for compatibility
    unk_token_id: int = 4
    ignore_token_id: int = 5

    # Noise schedule
    schedule_type: str = "cosine"  # "cosine" or "linear"

    # Noise injection parameters
    embed_noise_scale: float = 0.1  # Scale for Gaussian noise on embeddings
    swap_prob_scale: float = 0.95   # Probability of swapping embeddings at max noise


class SparseState:
    """
    Represents a sparse distribution over tokens at each position.

    Each position has k (probability, embedding, index) tuples.
    This is the core representation of Sparse Distribution Diffusion.

    Mathematical interpretation:
        position_i = Σ_{j=1}^{k} probs[i,j] × δ(embeds[i,j])

    where δ is the Dirac delta function in embedding space.

    Attributes:
        probs: [batch, seq_len, k] - probabilities (sorted descending)
        embeds: [batch, seq_len, k, embed_dim] - corresponding embeddings
        indices: [batch, seq_len, k] - token indices
    """

    def __init__(
        self,
        probs: torch.Tensor,
        embeds: torch.Tensor,
        indices: torch.Tensor,
    ):
        self.probs = probs
        self.embeds = embeds
        self.indices = indices

    @property
    def batch_size(self) -> int:
        return self.probs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.probs.shape[1]

    @property
    def k(self) -> int:
        return self.probs.shape[2]

    @property
    def embed_dim(self) -> int:
        return self.embeds.shape[3]

    @property
    def device(self) -> torch.device:
        return self.probs.device

    def top1_tokens(self) -> torch.Tensor:
        """Get the most likely token at each position."""
        return self.indices[:, :, 0]  # [batch, seq_len]

    def top1_probs(self) -> torch.Tensor:
        """Get the probability of the top token at each position."""
        return self.probs[:, :, 0]  # [batch, seq_len]

    def to(self, device: torch.device) -> "SparseState":
        """Move state to device."""
        return SparseState(
            probs=self.probs.to(device),
            embeds=self.embeds.to(device),
            indices=self.indices.to(device),
        )


class SparseDiffusion(nn.Module):
    """
    Sparse Distribution Diffusion process.

    Manages the noise injection (forward process) and provides utilities
    for the denoising (reverse process).

    Forward process (noise injection):
        1. Mix probabilities toward uniform (flatten distribution)
        2. Add Gaussian noise to embeddings (blur token meanings)
        3. Swap some embeddings with random ones (inject new candidates)

    Reverse process (denoising):
        1. Model predicts logits over full vocabulary
        2. Convert to top-k sparse state
        3. Repeat with decreasing noise level
    """

    def __init__(self, config: SDDConfig, embedding_table: nn.Embedding):
        super().__init__()
        self.config = config
        self.embedding_table = embedding_table  # Shared embedding table

    def get_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get noise level from timestep using schedule.

        Args:
            t: Timesteps in [0, 1] where 0=clean, 1=fully noisy

        Returns:
            noise_level in [0, 1]
        """
        if self.config.schedule_type == "cosine":
            # Cosine schedule: matches discrete diffusion
            # At t=0: rate=0, at t=1: rate=1
            # Uses quarter cosine for smoother endpoints
            return 1 - torch.cos(t * math.pi / 2)
        else:
            # Linear schedule
            return t

    def initialize_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> SparseState:
        """
        Initialize state with random uniform distribution.

        This is the "maximally noisy" state - pure uncertainty.
        No MASK token needed; we start from random embeddings.

        Args:
            batch_size: Number of sequences
            seq_len: Length of each sequence
            device: Device to create tensors on

        Returns:
            SparseState with random uniform distribution
        """
        k = self.config.k
        vocab_size = self.config.vocab_size

        # Sample k random tokens for each position
        indices = torch.randint(
            0, vocab_size,
            (batch_size, seq_len, k),
            device=device
        )

        # Look up their embeddings
        embeds = self.embedding_table(indices)  # [batch, seq, k, embed_dim]

        # Uniform probabilities
        probs = torch.ones(batch_size, seq_len, k, device=device) / k

        return SparseState(probs=probs, embeds=embeds, indices=indices)

    def add_noise(
        self,
        state: SparseState,
        t: torch.Tensor,
    ) -> SparseState:
        """
        Add noise to a sparse state (forward process).

        Noise is added in three ways:
        1. Mix probabilities toward uniform (flatten distribution)
        2. Add Gaussian noise to embeddings (blur token meanings)
        3. Swap some embeddings with random others (inject new candidates)

        Args:
            state: Clean or partially noisy SparseState
            t: Noise level in [0, 1] for each batch item

        Returns:
            Noisy SparseState
        """
        noise_level = self.get_noise_level(t)  # [batch]
        noise_level = noise_level.view(-1, 1, 1)  # [batch, 1, 1]

        batch_size, seq_len, k = state.probs.shape
        device = state.device

        # 1. Mix probabilities toward uniform
        uniform_prob = 1.0 / k
        noisy_probs = (1 - noise_level) * state.probs + noise_level * uniform_prob

        # 2. Add Gaussian noise to embeddings
        noise_level_4d = noise_level.unsqueeze(-1)  # [batch, 1, 1, 1]
        embed_noise = torch.randn_like(state.embeds) * self.config.embed_noise_scale
        noisy_embeds = state.embeds + noise_level_4d * embed_noise

        # 3. Swap some embeddings with random ones
        swap_prob = noise_level.squeeze(-1) * self.config.swap_prob_scale  # [batch, 1]
        swap_mask = torch.rand(batch_size, seq_len, k, device=device) < swap_prob.unsqueeze(-1)

        # Sample random indices for swapping
        random_indices = torch.randint(
            0, self.config.vocab_size,
            (batch_size, seq_len, k),
            device=device
        )
        random_embeds = self.embedding_table(random_indices)

        # Apply swaps
        noisy_embeds = torch.where(
            swap_mask.unsqueeze(-1),
            random_embeds,
            noisy_embeds
        )
        noisy_indices = torch.where(swap_mask, random_indices, state.indices)

        return SparseState(
            probs=noisy_probs,
            embeds=noisy_embeds,
            indices=noisy_indices,
        )

    def state_from_tokens(
        self,
        tokens: torch.Tensor,
        temperature: float = 0.01,
    ) -> SparseState:
        """
        Create a sparse state from discrete tokens (for training).

        Creates a peaked distribution around the ground truth token,
        with small probability mass on random other tokens.

        Args:
            tokens: Ground truth token indices [batch, seq_len]
            temperature: Controls sharpness (lower = more peaked)

        Returns:
            SparseState with peaked distribution on ground truth
        """
        batch_size, seq_len = tokens.shape
        k = self.config.k
        device = tokens.device

        # Ground truth token gets most of the probability
        gt_prob = 1.0 - temperature * (k - 1)
        other_prob = temperature

        # Build indices: ground truth first, then random others
        indices = torch.zeros(batch_size, seq_len, k, dtype=torch.long, device=device)
        indices[:, :, 0] = tokens  # Ground truth in first position

        # Fill remaining positions with random tokens
        random_tokens = torch.randint(
            0, self.config.vocab_size,
            (batch_size, seq_len, k - 1),
            device=device
        )
        indices[:, :, 1:] = random_tokens

        # Build probabilities: ground truth high, others low
        probs = torch.full((batch_size, seq_len, k), other_prob, device=device)
        probs[:, :, 0] = gt_prob

        # Get embeddings
        embeds = self.embedding_table(indices)

        return SparseState(probs=probs, embeds=embeds, indices=indices)

    def training_step(
        self,
        model: nn.Module,
        tokens: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step: add noise, predict, compute loss.

        Args:
            model: SparseDenoiser model
            tokens: Ground truth tokens [batch, seq_len]
            encoder_output: Optional conditioning (e.g., image features)
            encoder_mask: Optional mask for encoder output

        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Replace PAD tokens with IGNORE in targets
        # This teaches the model to predict IGNORE for "empty" positions
        targets = tokens.clone()
        targets[tokens == self.config.pad_token_id] = self.config.ignore_token_id

        # Create clean state from targets (with IGNORE instead of PAD)
        clean_state = self.state_from_tokens(targets)

        # Sample random timesteps
        t = torch.rand(batch_size, device=device)

        # Add noise
        noisy_state = self.add_noise(clean_state, t)

        # Get model predictions (logits)
        logits = model(
            noisy_state,
            t,
            encoder_output=encoder_output,
            encoder_mask=encoder_mask,
        )  # [batch, seq_len, vocab_size]

        # Compute weighted cross-entropy loss with noise-aware IGNORE weighting
        # At high noise (t≈1), focus on content; at low noise (t≈0), also learn IGNORE
        content_mask = (tokens != self.config.pad_token_id).float()  # [B, L]
        ignore_mask = (tokens == self.config.pad_token_id).float()   # [B, L]

        # Noise-aware weighting: IGNORE weight scales with (1-t)^2
        # At t=1: IGNORE weight ≈ 0 (don't penalize, just learn content)
        # At t=0: IGNORE weight = 1 (fully penalize)
        t_expanded = t.view(-1, 1)  # [B, 1]
        ignore_weight = (1 - t_expanded) ** 2  # [B, 1]

        # Content weight stays high (10x), IGNORE weight scales with noise level
        weights = content_mask * 10.0 + ignore_mask * ignore_weight * 1.0
        weights = weights.view(-1)

        # Per-token cross-entropy
        per_token_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            reduction='none',
        )

        # Weighted mean (add small epsilon to avoid division issues)
        loss = (per_token_loss * weights).sum() / (weights.sum() + 1e-8)

        # Compute metrics (accuracy on content tokens only, for comparability)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            content_mask = tokens != self.config.pad_token_id
            content_correct = (preds == targets) & content_mask
            content_accuracy = content_correct.float().sum() / content_mask.float().sum()

            # Also track IGNORE accuracy (how well model predicts IGNORE for padding)
            ignore_mask = tokens == self.config.pad_token_id
            ignore_correct = (preds == self.config.ignore_token_id) & ignore_mask
            ignore_accuracy = ignore_correct.float().sum() / (ignore_mask.float().sum() + 1e-8)

        metrics = {
            "loss": loss.item(),
            "accuracy": content_accuracy.item(),  # Content token accuracy
            "ignore_acc": ignore_accuracy.item(),  # IGNORE prediction accuracy
            "mean_t": t.mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        num_steps: int = 25,
        temperature: float = 1.0,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, SparseState]:
        """
        Generate sequences by iterative denoising (DDIM-style deterministic).

        Starting from pure noise (random uniform distribution), we
        iteratively refine the sparse state by:
        1. Running the model to get predicted logits
        2. Converting to top-k sparse state with temperature scaling
        3. Repeating until t=0 (no re-noising between steps)

        Args:
            model: SparseDenoiser model
            batch_size: Number of sequences to generate
            seq_len: Length of sequences
            num_steps: Number of denoising steps
            temperature: Sampling temperature (lower = more confident)
            encoder_output: Optional conditioning
            encoder_mask: Optional encoder mask
            device: Device to use

        Returns:
            tokens: Final discrete tokens [batch, seq_len]
            final_state: Final sparse state
        """
        model.eval()

        # Initialize with random uniform state
        state = self.initialize_state(batch_size, seq_len, device)

        # Move encoder outputs to device if provided
        if encoder_output is not None:
            encoder_output = encoder_output.to(device)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.to(device)

        # Timestep schedule: t goes from 1.0 to 0.0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        # Iterative denoising (DDIM-style: no re-noising)
        for i in range(num_steps):
            t_curr = timesteps[i].expand(batch_size)

            # Get model prediction with temperature
            state = model.denoise_step(
                state, t_curr,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
                temperature=temperature,
            )

            # NOTE: No re-noising between steps!
            # The model is trained to predict clean from noisy at level t.
            # Re-noising would destroy the model's predictions and cause
            # instability. This is DDIM-style deterministic sampling.

        # Final decoding: take top-1 tokens
        tokens = state.top1_tokens()

        return tokens, state

    def decode_with_ignore(
        self,
        tokens: torch.Tensor,
        tokenizer,
    ) -> list:
        """
        Decode tokens to text, removing IGNORE tokens.

        Args:
            tokens: Token indices [batch, seq_len]
            tokenizer: Tokenizer for decoding

        Returns:
            List of decoded strings
        """
        ignore_id = self.config.ignore_token_id
        pad_id = self.config.pad_token_id
        bos_id = self.config.bos_token_id
        eos_id = self.config.eos_token_id

        results = []
        for seq in tokens:
            # Filter out special tokens
            filtered = [
                t.item() for t in seq
                if t.item() not in [ignore_id, pad_id, bos_id, eos_id]
            ]
            text = tokenizer.decode(filtered)
            results.append(text)

        return results
