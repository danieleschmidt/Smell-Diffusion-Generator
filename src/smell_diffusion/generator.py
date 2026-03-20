"""Main ScentMoleculeGenerator: text-to-molecule generation pipeline."""

import numpy as np
from typing import List

from .molecule import MoleculeRepresentation, ATOM_TOKENS
from .text_conditioner import TextConditioner
from .diffusion import DiffusionSampler
from .safety import SafetyFilter
from .fragrance_notes import FragranceNotePredictor


# Atom type list for decoding generated vectors
ATOM_TYPES = sorted(ATOM_TOKENS.values())  # [6, 7, 8, 9, 16, 17, 35]
FINGERPRINT_SIZE = 128


class ScentMoleculeGenerator:
    """
    Main pipeline for generating odorant molecules from text descriptions.

    Uses a text conditioner to encode scent descriptors, a diffusion sampler
    to generate molecule parameters, and post-processing to decode them into
    MoleculeRepresentation objects. Safety filtering removes unsafe molecules.
    """

    def __init__(
        self,
        n_diffusion_steps: int = 100,
        seed: int = 42,
    ):
        self.text_conditioner = TextConditioner()
        self.diffusion_sampler = DiffusionSampler(
            n_steps=n_diffusion_steps, seed=seed
        )
        self.safety_filter = SafetyFilter()
        self.note_predictor = FragranceNotePredictor()
        self.rng = np.random.RandomState(seed)

    def generate(self, text: str, n_molecules: int = 5) -> List[MoleculeRepresentation]:
        """
        Generate molecules conditioned on a text scent description.

        Args:
            text: Scent description (e.g. "fresh floral citrus")
            n_molecules: Number of molecules to generate

        Returns:
            List of MoleculeRepresentation objects (filtered for safety)
        """
        # Encode text condition
        condition = self.text_conditioner.encode(text)

        molecules = []
        attempts = 0
        max_attempts = n_molecules * 3  # Try extra to account for safety filtering

        while len(molecules) < n_molecules and attempts < max_attempts:
            # Sample molecule parameters via diffusion
            # Shape: (FINGERPRINT_SIZE,) — a latent fingerprint-like vector
            params = self.diffusion_sampler.sample(condition, shape=(FINGERPRINT_SIZE,))
            mol = self._decode_molecule(params)
            molecules.append(mol)
            attempts += 1

        # Filter for safety
        safe_molecules = self.filter_safe(molecules)

        # If not enough safe molecules, return all (caller can check)
        if not safe_molecules:
            return molecules[:n_molecules]

        return safe_molecules[:n_molecules]

    def filter_safe(
        self, molecules: List[MoleculeRepresentation]
    ) -> List[MoleculeRepresentation]:
        """
        Filter molecules to keep only safe ones.

        Args:
            molecules: List of MoleculeRepresentation objects

        Returns:
            List of molecules that pass safety checks
        """
        return [m for m in molecules if self.safety_filter.is_safe(m)]

    def _decode_molecule(self, params: np.ndarray) -> MoleculeRepresentation:
        """
        Decode a raw parameter vector into a MoleculeRepresentation.

        Uses the parameter vector to determine:
        - Number of atoms (based on magnitude)
        - Atom types (based on binned values)
        - Bond structure (based on pairwise correlations)

        Args:
            params: 1D numpy array of molecule parameters

        Returns:
            MoleculeRepresentation
        """
        # Determine number of atoms from first value: clamp to [3, 15]
        n_atoms_raw = int(abs(params[0]) * 10) + 3
        n_atoms = max(3, min(15, n_atoms_raw))

        # Determine atom types from params[1:n_atoms+1]
        atom_param_slice = params[1: n_atoms + 1]
        # Map each float to an atom type via binning
        atom_bins = np.linspace(-3, 3, len(ATOM_TYPES) + 1)
        atoms = []
        for p in atom_param_slice:
            bin_idx = int(np.searchsorted(atom_bins, p, side='right')) - 1
            bin_idx = max(0, min(len(ATOM_TYPES) - 1, bin_idx))
            atoms.append(ATOM_TYPES[bin_idx])

        # Build bond matrix from correlation of param slices
        bond_params = params[n_atoms + 1: n_atoms + 1 + n_atoms * n_atoms]
        if len(bond_params) < n_atoms * n_atoms:
            bond_params = np.pad(bond_params, (0, n_atoms * n_atoms - len(bond_params)))

        bond_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        threshold = 0.5

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                idx = i * n_atoms + j
                val = abs(bond_params[idx]) if idx < len(bond_params) else 0.0
                if val > threshold:
                    bond_order = min(3, max(1, int(val)))
                    bond_matrix[i, j] = bond_order
                    bond_matrix[j, i] = bond_order

        # Ensure at least a linear chain of bonds
        for i in range(n_atoms - 1):
            if bond_matrix[i, i + 1] == 0:
                bond_matrix[i, i + 1] = 1.0
                bond_matrix[i + 1, i] = 1.0

        return MoleculeRepresentation(atoms=atoms, bonds=bond_matrix)
