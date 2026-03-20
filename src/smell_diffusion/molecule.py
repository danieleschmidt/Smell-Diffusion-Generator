"""Molecular representation for odorant molecules."""

import numpy as np
from typing import List, Optional


# Atom type constants: token -> atomic number
ATOM_TOKENS = {
    'C': 6,
    'O': 8,
    'N': 7,
    'S': 16,
    'F': 9,
    'Cl': 17,
    'Br': 35,
}

# Valence rules: atomic number -> max valence
VALENCE_RULES = {
    6: 4,   # Carbon
    8: 2,   # Oxygen
    7: 3,   # Nitrogen
    16: 6,  # Sulfur
    9: 1,   # Fluorine
    17: 1,  # Chlorine
    35: 1,  # Bromine
    1: 1,   # Hydrogen (implicit)
}

# Fingerprint length
FINGERPRINT_SIZE = 128


class MoleculeRepresentation:
    """
    Simplified molecular representation using atom types and adjacency matrix.

    Attributes:
        atoms: list of ints representing atom types (atomic numbers)
        bonds: numpy adjacency matrix of shape (n_atoms, n_atoms)
    """

    def __init__(self, atoms: List[int], bonds: Optional[np.ndarray] = None):
        self.atoms = list(atoms)
        n = len(atoms)
        if bonds is not None:
            self.bonds = np.array(bonds, dtype=np.float32)
        else:
            self.bonds = np.zeros((n, n), dtype=np.float32)

    @classmethod
    def from_smiles_like(cls, tokens: List[str]) -> "MoleculeRepresentation":
        """
        Parse a simplified SMILES-like token list into a MoleculeRepresentation.

        Tokens can be atom symbols (C, O, N, S, F, Cl, Br) or bond indicators
        ('-', '=', '#' for single, double, triple bonds). Atoms are connected
        linearly by default; branch notation '(' and ')' is supported.

        Example: ['C', 'C', 'O', '-', 'C'] or ['C', '(', 'C', ')', 'O']
        """
        atoms = []
        bonds_list = []
        # Stack for branch handling: stores atom index at branch start
        stack = []
        prev_idx = None
        pending_bond_order = 1
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token == '(':
                # Start branch - push current atom
                if prev_idx is not None:
                    stack.append(prev_idx)
                i += 1
                continue
            elif token == ')':
                # End branch - pop atom index
                if stack:
                    prev_idx = stack.pop()
                i += 1
                continue
            elif token == '-':
                pending_bond_order = 1
                i += 1
                continue
            elif token == '=':
                pending_bond_order = 2
                i += 1
                continue
            elif token == '#':
                pending_bond_order = 3
                i += 1
                continue
            elif token in ATOM_TOKENS:
                atom_num = ATOM_TOKENS[token]
                idx = len(atoms)
                atoms.append(atom_num)
                if prev_idx is not None:
                    bonds_list.append((prev_idx, idx, pending_bond_order))
                prev_idx = idx
                pending_bond_order = 1
            # Unknown tokens are skipped
            i += 1

        n = len(atoms)
        bond_matrix = np.zeros((n, n), dtype=np.float32)
        for a, b, order in bonds_list:
            bond_matrix[a, b] = order
            bond_matrix[b, a] = order

        return cls(atoms=atoms, bonds=bond_matrix)

    def to_fingerprint(self) -> np.ndarray:
        """
        Compute a Morgan-like circular fingerprint as a 128-dim numpy vector.

        Uses atom types, neighbor environments, and bond orders as features,
        hashed into a fixed-length bit vector.
        """
        fp = np.zeros(FINGERPRINT_SIZE, dtype=np.float32)
        n = len(self.atoms)

        if n == 0:
            return fp

        # Radius-0: atom identity
        for i, atom in enumerate(self.atoms):
            idx = atom % FINGERPRINT_SIZE
            fp[idx] += 1.0

        # Radius-1: atom + neighbor environment
        for i in range(n):
            neighbors = np.where(self.bonds[i] > 0)[0]
            neighbor_sum = sum(self.atoms[j] for j in neighbors)
            bond_sum = int(np.sum(self.bonds[i]))
            env_hash = (self.atoms[i] * 31 + neighbor_sum * 17 + bond_sum * 7) % FINGERPRINT_SIZE
            fp[env_hash] += 1.0

        # Radius-2: atom + 2-hop environment
        for i in range(n):
            neighbors_1 = np.where(self.bonds[i] > 0)[0]
            two_hop_sum = 0
            for j in neighbors_1:
                neighbors_2 = np.where(self.bonds[j] > 0)[0]
                two_hop_sum += sum(self.atoms[k] for k in neighbors_2 if k != i)
            env_hash = (self.atoms[i] * 53 + two_hop_sum * 29) % FINGERPRINT_SIZE
            fp[env_hash] += 1.0

        # Normalize to [0, 1]
        max_val = fp.max()
        if max_val > 0:
            fp = fp / max_val

        return fp

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        """Number of bonds (edges) in the molecule."""
        # Each bond is symmetric in the matrix; count upper triangle
        return int(np.sum(self.bonds > 0)) // 2

    def validate(self) -> bool:
        """
        Validate molecule against valence rules.

        Returns True if all atoms satisfy their valence constraints.
        """
        warnings = self._get_valence_warnings()
        return len(warnings) == 0

    def _get_valence_warnings(self) -> List[str]:
        """Return list of valence violation messages."""
        warnings = []
        for i, atom in enumerate(self.atoms):
            max_valence = VALENCE_RULES.get(atom, 4)
            actual_valence = int(np.sum(self.bonds[i]))
            if actual_valence > max_valence:
                warnings.append(
                    f"Atom {i} (atomic num {atom}) has valence {actual_valence}, "
                    f"exceeds max {max_valence}"
                )
        return warnings

    def __repr__(self) -> str:
        return (
            f"MoleculeRepresentation(n_atoms={self.num_atoms}, "
            f"n_bonds={self.num_bonds}, atoms={self.atoms})"
        )
