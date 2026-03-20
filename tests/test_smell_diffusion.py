"""Tests for the Smell-Diffusion-Generator package."""

import numpy as np
import pytest
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smell_diffusion.molecule import MoleculeRepresentation
from smell_diffusion.text_conditioner import TextConditioner
from smell_diffusion.diffusion import DiffusionSampler
from smell_diffusion.safety import SafetyFilter
from smell_diffusion.fragrance_notes import FragranceNotePredictor
from smell_diffusion.generator import ScentMoleculeGenerator


# ---------------------------------------------------------------------------
# MoleculeRepresentation tests
# ---------------------------------------------------------------------------

def test_molecule_representation_basic():
    """Test basic molecule construction."""
    atoms = [6, 6, 8]  # C, C, O
    bonds = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)
    mol = MoleculeRepresentation(atoms=atoms, bonds=bonds)
    assert mol.num_atoms == 3
    assert mol.num_bonds == 2
    assert mol.atoms == [6, 6, 8]


def test_molecule_fingerprint_shape():
    """Test that fingerprint returns a 128-dim vector."""
    atoms = [6, 6, 8, 7]
    bonds = np.zeros((4, 4), dtype=np.float32)
    bonds[0, 1] = bonds[1, 0] = 1
    bonds[1, 2] = bonds[2, 1] = 1
    bonds[2, 3] = bonds[3, 2] = 2
    mol = MoleculeRepresentation(atoms=atoms, bonds=bonds)
    fp = mol.to_fingerprint()
    assert fp.shape == (128,)
    assert fp.dtype == np.float32
    assert np.all(fp >= 0.0)
    assert np.all(fp <= 1.0)


def test_molecule_validate():
    """Test molecule valence validation."""
    # Valid molecule: C-C-O (valences: C=4, O=2)
    atoms = [6, 6, 8]
    bonds = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)
    mol_valid = MoleculeRepresentation(atoms=atoms, bonds=bonds)
    assert mol_valid.validate() is True

    # Invalid: Oxygen with 4 bonds (exceeds valence=2)
    atoms_bad = [8]
    bonds_bad = np.array([[4.0]], dtype=np.float32)
    mol_bad = MoleculeRepresentation(atoms=atoms_bad, bonds=bonds_bad)
    assert mol_bad.validate() is False


def test_molecule_from_smiles_like():
    """Test parsing simplified SMILES-like token list."""
    tokens = ['C', 'C', 'O', 'C']
    mol = MoleculeRepresentation.from_smiles_like(tokens)
    assert mol.num_atoms == 4
    assert mol.atoms == [6, 6, 8, 6]
    assert mol.num_bonds == 3


def test_molecule_from_smiles_like_branches():
    """Test SMILES-like parsing with branch notation."""
    tokens = ['C', '(', 'C', ')', 'O']
    mol = MoleculeRepresentation.from_smiles_like(tokens)
    assert mol.num_atoms == 3
    # C branches to C, then continues to O: C(-C)-O


# ---------------------------------------------------------------------------
# TextConditioner tests
# ---------------------------------------------------------------------------

def test_text_conditioner_encode_floral():
    """Test encoding of known scent descriptor."""
    tc = TextConditioner()
    vec = tc.encode("floral")
    assert vec.shape == (64,)
    assert vec.dtype == np.float32
    # Should not be zero for a known descriptor
    assert np.linalg.norm(vec) > 0.1


def test_text_conditioner_encode_unknown():
    """Test encoding of unknown text returns zero vector."""
    tc = TextConditioner()
    vec = tc.encode("xyzzyblorp quantum foam")
    assert vec.shape == (64,)
    assert np.allclose(vec, 0.0)


def test_text_conditioner_batch_encode():
    """Test batch encoding returns correct shape."""
    tc = TextConditioner()
    texts = ["floral citrus", "woody musky", "fresh clean"]
    result = tc.batch_encode(texts)
    assert result.shape == (3, 64)
    assert result.dtype == np.float32


def test_text_conditioner_different_descriptors_differ():
    """Test that different descriptors produce different vectors."""
    tc = TextConditioner()
    v_floral = tc.encode("floral")
    v_woody = tc.encode("woody")
    assert not np.allclose(v_floral, v_woody)


# ---------------------------------------------------------------------------
# DiffusionSampler tests
# ---------------------------------------------------------------------------

def test_diffusion_noise_schedule():
    """Test that diffusion schedules are computed correctly."""
    ds = DiffusionSampler(n_steps=100, beta_start=0.0001, beta_end=0.02)
    assert len(ds.betas) == 100
    assert len(ds.alphas) == 100
    assert len(ds.alpha_bars) == 100
    # Betas should be monotonically increasing (linear schedule)
    assert ds.betas[0] < ds.betas[-1]
    # Alpha bars should be monotonically decreasing
    assert ds.alpha_bars[0] > ds.alpha_bars[-1]
    # All alpha bars should be in (0, 1]
    assert np.all(ds.alpha_bars > 0)
    assert np.all(ds.alpha_bars <= 1.0)


def test_diffusion_add_noise():
    """Test that add_noise increases noise level at higher t."""
    ds = DiffusionSampler(n_steps=100, seed=1)
    x = np.ones(16, dtype=np.float32)
    x_t_low, eps_low = ds.add_noise(x, t=0)
    x_t_high, eps_high = ds.add_noise(x, t=99)
    assert x_t_low.shape == (16,)
    assert x_t_high.shape == (16,)
    # At t=99, signal should be almost entirely noise
    # alpha_bar[99] should be very small, so signal is minimal
    assert ds.alpha_bars[99] < ds.alpha_bars[0]


def test_diffusion_sample_shape():
    """Test that sample returns the correct shape."""
    ds = DiffusionSampler(n_steps=10, seed=0)  # Fewer steps for speed
    condition = np.zeros(64, dtype=np.float32)
    shape = (32,)
    result = ds.sample(condition, shape)
    assert result.shape == shape
    assert result.dtype == np.float32


def test_diffusion_denoise_step():
    """Test a single denoising step returns correct shape."""
    ds = DiffusionSampler(n_steps=10, seed=0)
    x_t = np.random.randn(32).astype(np.float32)
    condition = np.random.randn(64).astype(np.float32)
    x_prev = ds.denoise_step(x_t, t=5, condition=condition)
    assert x_prev.shape == (32,)


# ---------------------------------------------------------------------------
# SafetyFilter tests
# ---------------------------------------------------------------------------

def test_safety_filter_basic():
    """Test that a normal small molecule is considered safe."""
    sf = SafetyFilter()
    atoms = [6, 6, 8, 6, 6]  # Normal C-C-O-C-C
    bonds = np.zeros((5, 5), dtype=np.float32)
    for i in range(4):
        bonds[i, i+1] = bonds[i+1, i] = 1.0
    mol = MoleculeRepresentation(atoms=atoms, bonds=bonds)
    assert sf.is_safe(mol) is True
    assert sf.get_warnings(mol) == []


def test_safety_filter_large_molecule():
    """Test that an oversized molecule triggers a warning."""
    sf = SafetyFilter()
    # Create molecule with 60 atoms (exceeds max_atoms=50)
    atoms = [6] * 60
    bonds = np.zeros((60, 60), dtype=np.float32)
    mol = MoleculeRepresentation(atoms=atoms, bonds=bonds)
    assert sf.is_safe(mol) is False
    warnings = sf.get_warnings(mol)
    assert any("atoms" in w for w in warnings)


def test_safety_filter_bromine_warning():
    """Test that too many bromine atoms triggers a warning."""
    sf = SafetyFilter()
    atoms = [35, 35, 35, 6, 6]  # 3 Br atoms (exceeds max 2)
    bonds = np.zeros((5, 5), dtype=np.float32)
    mol = MoleculeRepresentation(atoms=atoms, bonds=bonds)
    assert sf.is_safe(mol) is False
    warnings = sf.get_warnings(mol)
    assert any("Bromine" in w or "bromine" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# FragranceNotePredictor tests
# ---------------------------------------------------------------------------

def test_fragrance_note_predictor():
    """Test fragrance note prediction returns valid probabilities."""
    fnp = FragranceNotePredictor()

    # Small molecule → should favor top note
    small_mol = MoleculeRepresentation(atoms=[6, 9, 9], bonds=np.zeros((3, 3)))
    probs_small = fnp.predict(small_mol)
    assert set(probs_small.keys()) == {"top_note", "middle_note", "base_note"}
    assert abs(sum(probs_small.values()) - 1.0) < 1e-5
    assert all(v >= 0 for v in probs_small.values())

    # Large molecule → should favor base note
    large_atoms = [6, 16, 35, 16, 35] * 5  # 25 atoms with heavy elements
    large_mol = MoleculeRepresentation(atoms=large_atoms, bonds=np.zeros((25, 25)))
    probs_large = fnp.predict(large_mol)
    assert set(probs_large.keys()) == {"top_note", "middle_note", "base_note"}


def test_fragrance_note_classify():
    """Test that classify returns a valid note label."""
    fnp = FragranceNotePredictor()
    atoms = [6, 6, 8]
    mol = MoleculeRepresentation(atoms=atoms, bonds=np.zeros((3, 3)))
    label = fnp.classify(mol)
    assert label in {"top", "middle", "base"}


# ---------------------------------------------------------------------------
# Generator end-to-end test
# ---------------------------------------------------------------------------

def test_generator_end_to_end():
    """Test full generation pipeline produces valid molecules."""
    gen = ScentMoleculeGenerator(n_diffusion_steps=10, seed=0)
    molecules = gen.generate("fresh floral citrus", n_molecules=3)
    # Should return at least 1 molecule
    assert len(molecules) >= 1
    for mol in molecules:
        assert isinstance(mol, MoleculeRepresentation)
        assert mol.num_atoms >= 1
        fp = mol.to_fingerprint()
        assert fp.shape == (128,)


def test_generator_filter_safe():
    """Test that filter_safe removes unsafe molecules."""
    gen = ScentMoleculeGenerator(seed=0)
    # Create a mix of safe and unsafe molecules
    safe_mol = MoleculeRepresentation(atoms=[6, 6, 8], bonds=np.zeros((3, 3)))
    unsafe_mol = MoleculeRepresentation(atoms=[6] * 60, bonds=np.zeros((60, 60)))
    filtered = gen.filter_safe([safe_mol, unsafe_mol])
    assert safe_mol in filtered
    assert unsafe_mol not in filtered
