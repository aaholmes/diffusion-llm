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
- State: (probs, indices) representing top-k tokens per position
- Noise: Mix toward uniform + probability-based swapping
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

    # Noise injection parameters removed - using probability-based swapping instead


class SparseState:
    """
    Represents a sparse distribution over tokens at each position.

    Each position has k (probability, index) tuples. Embeddings are looked up
    from the model's embedding table when needed, not stored in the state.

    Mathematical interpretation:
        position_i = Σ_{j=1}^{k} probs[i,j] × δ(embedding[indices[i,j]])

    where δ is the Dirac delta function in embedding space.

    Attributes:
        probs: [batch, seq_len, k] - probabilities (sorted descending)
        indices: [batch, seq_len, k] - token indices
    """

    def __init__(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ):
        self.probs = probs
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
            indices=self.indices.to(device),
        )


class SparseDiffusion(nn.Module):
    """
    Sparse Distribution Diffusion process.

    Manages the noise injection (forward process) and provides utilities
    for the denoising (reverse process).

    Forward process (noise injection):
        1. Compute swap probability based on ORIGINAL probs (before flattening)
           - High original prob → low swap chance
           - Low original prob → high swap chance
        2. Swap candidates based on swap probability
        3. Mix probabilities toward uniform (flatten distribution)
        4. Swapped tokens get 1/vocab_size probability (maximum uncertainty)

    This probability-based swapping ensures training distribution matches
    inference distribution at t=1: all candidates have equal chance of being
    swapped, making training closer to random initialization.

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
        No MASK token needed; we start from random tokens.

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

        # Uniform probabilities (coarse-grained: each token has 1/vocab_size probability,
        # representing maximum uncertainty over the full vocabulary)
        probs = torch.ones(batch_size, seq_len, k, device=device) / vocab_size

        return SparseState(probs=probs, indices=indices)

    def add_noise(
        self,
        state: SparseState,
        t: torch.Tensor,
    ) -> SparseState:
        """
        Add noise to a sparse state (forward process).

        Uses probability-based swapping to ensure training distribution matches
        inference distribution at t=1:
        1. Compute swap probability based on ORIGINAL probs (before flattening)
           - swap_prob[i] = noise_level * (1 - normalized_prob[i])
           - High-prob candidates rarely swapped, low-prob candidates often swapped
        2. Swap candidates to random tokens based on swap probability
        3. Mix probabilities toward uniform (flatten distribution)
        4. Swapped tokens get 1/vocab_size probability (maximum uncertainty)

        At t≈0: High-prob candidate (GT) rarely swapped, signal preserved
        At t≈1: All probs are ~1/V, so all candidates equally likely to swap

        Args:
            state: Clean or partially noisy SparseState
            t: Noise level in [0, 1] for each batch item

        Returns:
            Noisy SparseState
        """
        noise_level = self.get_noise_level(t)  # [batch]
        device = state.probs.device

        # 1. Compute swap probability based on ORIGINAL probs (before flattening)
        # Normalize probs to [0,1] range for swap calculation
        # High original prob → low swap chance, low original prob → high swap chance
        prob_max = state.probs.max(dim=-1, keepdim=True).values  # [B, L, 1]
        normalized_probs = state.probs / (prob_max + 1e-8)  # [B, L, k] in [0, 1]

        # swap_prob = noise_level * (1 - normalized_probs * (1 - noise_level))
        # At t=0: swap_prob = 0 for all (no swaps)
        # At t=1: swap_prob = 1 for all (all candidates equally likely to swap)
        # At intermediate t: high-prob candidates swapped less than low-prob
        noise_3d = noise_level.view(-1, 1, 1)  # [B, 1, 1]
        swap_prob = noise_3d * (1 - normalized_probs * (1 - noise_3d))  # [B, L, k]
        swap_mask = torch.rand_like(swap_prob) < swap_prob  # [B, L, k]

        # 2. Generate random replacement indices for swapped positions
        random_indices = torch.randint(
            0, self.config.vocab_size,
            state.indices.shape,
            device=device
        )

        # 3. Apply swaps to indices
        new_indices = torch.where(swap_mask, random_indices, state.indices)

        # 4. Flatten probabilities toward uniform
        uniform_prob = 1.0 / self.config.vocab_size
        noisy_probs = (1 - noise_3d) * state.probs + noise_3d * uniform_prob

        # 5. Swapped tokens get 1/vocab_size probability (maximum uncertainty)
        noisy_probs = torch.where(swap_mask, uniform_prob, noisy_probs)

        return SparseState(
            probs=noisy_probs,
            indices=new_indices,
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

        return SparseState(probs=probs, indices=indices)

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
