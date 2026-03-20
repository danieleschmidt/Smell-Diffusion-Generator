"""Fragrance note classification (top/middle/base) for molecules."""

from typing import Dict
from .molecule import MoleculeRepresentation


class FragranceNotePredictor:
    """
    Classifies molecules into fragrance note categories based on
    structural characteristics related to volatility.

    - Top notes: small, volatile molecules (low atom count, high halogen)
    - Middle notes: medium-sized molecules (moderate complexity)
    - Base notes: large, heavy molecules (high atom count, ring-like structures)
    """

    # Atomic numbers considered high-volatility (halogens make molecules more volatile)
    TOP_NOTE_ATOMS = {
        "max_atom_count": 10,
        "volatile_types": {9, 17},   # F, Cl — increase volatility
        "min_volatile_fraction": 0.1,
    }

    MIDDLE_NOTE_ATOMS = {
        "min_atom_count": 8,
        "max_atom_count": 25,
        "moderate_types": {6, 8, 7},  # C, O, N — typical mid-note building blocks
    }

    BASE_NOTE_ATOMS = {
        "min_atom_count": 20,
        "heavy_types": {16, 35},      # S, Br — heavy/anchoring atoms
        "ring_proxy_threshold": 3,    # 3+ symmetrical bonds as ring proxy
    }

    def predict(self, molecule: MoleculeRepresentation) -> Dict[str, float]:
        """
        Predict probabilities for top, middle, and base note classification.

        Uses heuristic scoring based on molecule size and atom composition.
        Probabilities sum to 1.0.

        Args:
            molecule: MoleculeRepresentation to classify

        Returns:
            dict with keys 'top_note', 'middle_note', 'base_note' as floats
        """
        n = molecule.num_atoms
        atoms = molecule.atoms

        if n == 0:
            return {"top_note": 1/3, "middle_note": 1/3, "base_note": 1/3}

        # Atom type counts
        volatile_count = sum(1 for a in atoms if a in self.TOP_NOTE_ATOMS["volatile_types"])
        heavy_count = sum(1 for a in atoms if a in self.BASE_NOTE_ATOMS["heavy_types"])
        volatile_fraction = volatile_count / n
        heavy_fraction = heavy_count / n

        # Ring proxy: count atoms with 3+ bonds (degree >= 3 suggests ring membership)
        import numpy as np
        ring_atoms = int(np.sum(np.sum(molecule.bonds > 0, axis=1) >= 3)) if n > 0 else 0

        # Score each note
        # Top note score: small size, volatile atoms
        top_score = 0.0
        if n <= self.TOP_NOTE_ATOMS["max_atom_count"]:
            top_score += 1.0 - (n / self.TOP_NOTE_ATOMS["max_atom_count"])
        top_score += volatile_fraction * 2.0

        # Middle note score: moderate size
        mid_score = 0.0
        mid_min = self.MIDDLE_NOTE_ATOMS["min_atom_count"]
        mid_max = self.MIDDLE_NOTE_ATOMS["max_atom_count"]
        if mid_min <= n <= mid_max:
            # Peak at center of range
            center = (mid_min + mid_max) / 2
            mid_score += 1.0 - abs(n - center) / ((mid_max - mid_min) / 2)
        mid_score += sum(1 for a in atoms if a in self.MIDDLE_NOTE_ATOMS["moderate_types"]) / max(n, 1)

        # Base note score: large size, heavy atoms, ring structures
        base_score = 0.0
        if n >= self.BASE_NOTE_ATOMS["min_atom_count"]:
            base_score += min(1.0, n / 40.0)
        base_score += heavy_fraction * 2.0
        if ring_atoms >= self.BASE_NOTE_ATOMS["ring_proxy_threshold"]:
            base_score += 0.5

        # Normalize to probabilities
        total = top_score + mid_score + base_score
        if total < 1e-8:
            return {"top_note": 1/3, "middle_note": 1/3, "base_note": 1/3}

        return {
            "top_note": float(top_score / total),
            "middle_note": float(mid_score / total),
            "base_note": float(base_score / total),
        }

    def classify(self, molecule: MoleculeRepresentation) -> str:
        """
        Classify molecule into the most likely fragrance note category.

        Args:
            molecule: MoleculeRepresentation to classify

        Returns:
            One of: "top", "middle", "base"
        """
        probs = self.predict(molecule)
        label_map = {
            "top_note": "top",
            "middle_note": "middle",
            "base_note": "base",
        }
        best = max(probs, key=probs.__getitem__)
        return label_map[best]
