"""Diffusion sampler for molecule parameter generation."""

import numpy as np
from typing import Tuple


class DiffusionSampler:
    """
    DDPM-style diffusion sampler using numpy only.

    Uses a linear beta schedule to define the forward noising process,
    and a learned score function (simple MLP via matmul) for the reverse process.
    """

    def __init__(
        self,
        n_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        seed: int = 0,
    ):
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.rng = np.random.RandomState(seed)

        # Linear beta schedule
        self.betas = np.linspace(beta_start, beta_end, n_steps, dtype=np.float32)

        # Compute alphas and cumulative products
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas).astype(np.float32)

        # Pre-build simple MLP weights for the score function
        # Input: x (shape[-1]) + t_emb (8) + condition (64) -> hidden (64) -> output (x_dim)
        self._init_score_weights()

    def _init_score_weights(self):
        """Initialize fixed MLP-like weights for the score function."""
        rng = np.random.RandomState(12345)
        # We'll build weights lazily on first use since x_dim is unknown at init
        self._score_weights = None

    def _build_score_weights(self, x_dim: int):
        """Build MLP weights for given input dimension."""
        rng = np.random.RandomState(12345)
        t_emb_dim = 8
        cond_dim = 64
        hidden_dim = 64
        input_dim = x_dim + t_emb_dim + cond_dim

        scale = 0.01
        self._score_weights = {
            'W1': rng.randn(input_dim, hidden_dim).astype(np.float32) * scale,
            'b1': np.zeros(hidden_dim, dtype=np.float32),
            'W2': rng.randn(hidden_dim, x_dim).astype(np.float32) * scale,
            'b2': np.zeros(x_dim, dtype=np.float32),
        }
        self._score_x_dim = x_dim

    def _time_embedding(self, t: int) -> np.ndarray:
        """Simple sinusoidal time embedding of dim 8."""
        dim = 8
        freqs = np.array([10000 ** (-2 * i / dim) for i in range(dim // 2)], dtype=np.float32)
        emb = np.zeros(dim, dtype=np.float32)
        emb[:dim // 2] = np.sin(t * freqs)
        emb[dim // 2:] = np.cos(t * freqs)
        return emb

    def add_noise(self, x: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward process: add noise to x at timestep t.

        x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * eps

        Args:
            x: Clean sample, shape (D,) or (B, D)
            t: Timestep index (0-indexed)

        Returns:
            (x_t, eps): noisy sample and the noise added
        """
        alpha_bar_t = self.alpha_bars[t]
        eps = self.rng.randn(*x.shape).astype(np.float32)
        x_t = np.sqrt(alpha_bar_t) * x + np.sqrt(1.0 - alpha_bar_t) * eps
        return x_t.astype(np.float32), eps

    def score_fn(
        self, x: np.ndarray, t: int, condition: np.ndarray
    ) -> np.ndarray:
        """
        Simple MLP-based score (noise prediction) function.

        Args:
            x: Noisy sample, shape (D,)
            t: Timestep index
            condition: Conditioning vector, shape (64,)

        Returns:
            Predicted noise, shape (D,)
        """
        x_dim = x.shape[-1]

        # Build weights lazily
        if self._score_weights is None or self._score_x_dim != x_dim:
            self._build_score_weights(x_dim)

        t_emb = self._time_embedding(t)

        # Concatenate features
        cond = condition[:64] if len(condition) >= 64 else np.pad(
            condition, (0, 64 - len(condition))
        )
        inp = np.concatenate([x.flatten(), t_emb, cond]).astype(np.float32)

        # Layer 1: linear + relu
        h = inp @ self._score_weights['W1'] + self._score_weights['b1']
        h = np.maximum(h, 0)  # ReLU

        # Layer 2: linear
        out = h @ self._score_weights['W2'] + self._score_weights['b2']
        return out.astype(np.float32)

    def denoise_step(
        self, x_t: np.ndarray, t: int, condition: np.ndarray
    ) -> np.ndarray:
        """
        Single reverse denoising step (DDPM reverse process).

        Args:
            x_t: Noisy sample at step t, shape (D,)
            t: Current timestep (0-indexed, counting down)
            condition: Conditioning vector

        Returns:
            x_{t-1}: Less noisy sample
        """
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        # Predict noise
        eps_pred = self.score_fn(x_t, t, condition)

        # DDPM reverse step mean
        coef = beta_t / np.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / np.sqrt(alpha_t)) * (x_t - coef * eps_pred)

        if t == 0:
            return mean.astype(np.float32)
        else:
            noise = self.rng.randn(*x_t.shape).astype(np.float32)
            sigma = np.sqrt(beta_t)
            return (mean + sigma * noise).astype(np.float32)

    def sample(
        self, condition: np.ndarray, shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Full reverse diffusion sampling loop.

        Starts from pure noise and iteratively denoises conditioned on
        the provided latent vector.

        Args:
            condition: Conditioning vector (e.g. from TextConditioner), shape (64,)
            shape: Shape of sample to generate, e.g. (128,)

        Returns:
            Denoised sample of given shape
        """
        # Start from pure noise
        x = self.rng.randn(*shape).astype(np.float32)

        # Reverse loop: T-1 down to 0
        for t in reversed(range(self.n_steps)):
            x = self.denoise_step(x, t, condition)

        return x
