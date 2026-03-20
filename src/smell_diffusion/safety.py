"""Safety filtering for generated molecules."""

from typing import List, Set, Dict, Any
from .molecule import MoleculeRepresentation


class SafetyFilter:
    """
    Safety filter for odorant molecules.

    Checks molecules against irritancy rules and known problematic structures.
    """

    # Safety rules: each rule defines a constraint
    IRRITANT_RULES: Dict[str, Any] = {
        "max_atoms": 50,
        "max_molecular_weight_proxy": 500,  # proxy: sum of atomic numbers
        "forbidden_atom_types": {35},       # Bromine (Br) in large quantities
        "max_bromine_count": 2,
        "max_fluorine_count": 5,
        "min_atoms": 1,
    }

    # Known irritant signatures: frozensets of atom-count tuples that are irritating
    # Format: frozenset of (atomic_number, min_count) pairs
    KNOWN_IRRITANTS: Set[frozenset] = {
        # Very high halogen content patterns
        frozenset({(9, 4), (17, 3)}),    # 4+ F and 3+ Cl atoms
        frozenset({(35, 3)}),             # 3+ Br atoms (highly irritating)
        frozenset({(17, 5)}),             # 5+ Cl atoms
        frozenset({(16, 4)}),             # 4+ S atoms (sulfurous/irritant)
    }

    def is_safe(self, molecule: MoleculeRepresentation) -> bool:
        """
        Determine if a molecule passes all safety checks.

        Args:
            molecule: MoleculeRepresentation to check

        Returns:
            True if molecule is considered safe, False otherwise
        """
        return len(self.get_warnings(molecule)) == 0

    def get_warnings(self, molecule: MoleculeRepresentation) -> List[str]:
        """
        Get list of safety warnings for a molecule.

        Args:
            molecule: MoleculeRepresentation to check

        Returns:
            List of warning strings (empty if safe)
        """
        warnings = []
        rules = self.IRRITANT_RULES
        atoms = molecule.atoms

        # Check minimum atoms
        if molecule.num_atoms < rules["min_atoms"]:
            warnings.append(
                f"Molecule has too few atoms ({molecule.num_atoms}); "
                f"minimum is {rules['min_atoms']}"
            )

        # Check max atoms
        if molecule.num_atoms > rules["max_atoms"]:
            warnings.append(
                f"Molecule has {molecule.num_atoms} atoms, "
                f"exceeding max {rules['max_atoms']}"
            )

        # Check molecular weight proxy (sum of atomic numbers)
        mw_proxy = sum(atoms)
        if mw_proxy > rules["max_molecular_weight_proxy"]:
            warnings.append(
                f"Molecular weight proxy ({mw_proxy}) exceeds "
                f"max {rules['max_molecular_weight_proxy']}"
            )

        # Check bromine count
        br_count = atoms.count(35)
        if br_count > rules["max_bromine_count"]:
            warnings.append(
                f"Too many Bromine atoms ({br_count}); "
                f"max allowed is {rules['max_bromine_count']}"
            )

        # Check fluorine count
        f_count = atoms.count(9)
        if f_count > rules["max_fluorine_count"]:
            warnings.append(
                f"Too many Fluorine atoms ({f_count}); "
                f"max allowed is {rules['max_fluorine_count']}"
            )

        # Check known irritant signatures
        atom_counts: Dict[int, int] = {}
        for a in atoms:
            atom_counts[a] = atom_counts.get(a, 0) + 1

        for signature in self.KNOWN_IRRITANTS:
            # Signature is a frozenset of (atomic_number, min_count) pairs
            matches = all(
                atom_counts.get(atomic_num, 0) >= min_count
                for atomic_num, min_count in signature
            )
            if matches:
                sig_str = ", ".join(
                    f"atom {an} x{mc}" for an, mc in sorted(signature)
                )
                warnings.append(f"Matches known irritant pattern: {sig_str}")

        return warnings
