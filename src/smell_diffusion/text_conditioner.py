"""Text conditioning for scent descriptor encoding."""

import numpy as np
from typing import List, Dict


# Latent vector dimensionality
LATENT_DIM = 64

# Seed for reproducible random vectors
_RNG = np.random.RandomState(42)


def _make_vec(seed: int) -> np.ndarray:
    """Generate a deterministic unit vector for a scent descriptor."""
    rng = np.random.RandomState(seed)
    v = rng.randn(LATENT_DIM).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


class TextConditioner:
    """
    Encodes scent descriptors from natural language text into latent vectors.

    The SCENT_VOCAB maps descriptor keywords to fixed 64-dim latent vectors.
    Unknown words are ignored; if no known words are found, a zero vector is returned.
    """

    SCENT_VOCAB: Dict[str, np.ndarray] = {
        "floral":  _make_vec(1),
        "citrus":  _make_vec(2),
        "woody":   _make_vec(3),
        "musky":   _make_vec(4),
        "fresh":   _make_vec(5),
        "sweet":   _make_vec(6),
        "spicy":   _make_vec(7),
        "earthy":  _make_vec(8),
        # Aliases and related terms
        "rose":    _make_vec(1),
        "jasmine": _make_vec(1),
        "lemon":   _make_vec(2),
        "orange":  _make_vec(2),
        "lime":    _make_vec(2),
        "cedar":   _make_vec(3),
        "sandalwood": _make_vec(3),
        "musk":    _make_vec(4),
        "clean":   _make_vec(5),
        "aquatic": _make_vec(5),
        "honey":   _make_vec(6),
        "vanilla": _make_vec(6),
        "pepper":  _make_vec(7),
        "cinnamon": _make_vec(7),
        "soil":    _make_vec(8),
        "moss":    _make_vec(8),
    }

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a text description into a 64-dim latent vector.

        Tokenizes text by whitespace and punctuation, finds descriptor words
        in SCENT_VOCAB, and returns their average embedding.
        If no known words are found, returns a zero vector.

        Args:
            text: Natural language description (e.g. "fresh floral citrus")

        Returns:
            np.ndarray of shape (64,)
        """
        tokens = self._tokenize(text)
        vecs = []
        for token in tokens:
            if token in self.SCENT_VOCAB:
                vecs.append(self.SCENT_VOCAB[token])

        if not vecs:
            return np.zeros(LATENT_DIM, dtype=np.float32)

        result = np.mean(vecs, axis=0).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result = result / norm
        return result

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text descriptions.

        Args:
            texts: List of description strings

        Returns:
            np.ndarray of shape (len(texts), 64)
        """
        return np.stack([self.encode(t) for t in texts], axis=0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase words, stripping punctuation."""
        import re
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return tokens
