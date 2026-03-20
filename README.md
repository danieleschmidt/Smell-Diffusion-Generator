# Smell-Diffusion-Generator

Cross-modal diffusion model generating odorant molecules from text descriptions.

## Features

- Text-conditioned molecule generation using diffusion sampling
- SMILES-inspired molecular representation
- Safety filtering for known irritants
- Fragrance note classification (top/middle/base)

## Usage

```python
from smell_diffusion.generator import ScentMoleculeGenerator

gen = ScentMoleculeGenerator()
molecules = gen.generate("fresh floral citrus")
```

## Install

```bash
pip install -e .
```

## Architecture

The pipeline consists of:

1. **TextConditioner** — encodes scent descriptors (floral, citrus, woody, etc.) into 64-dim latent vectors
2. **DiffusionSampler** — DDPM-style reverse diffusion (numpy-only) conditioned on the text embedding
3. **MoleculeRepresentation** — SMILES-inspired atom/bond representation with Morgan-like circular fingerprints
4. **SafetyFilter** — checks molecules against irritancy rules and known problematic patterns
5. **FragranceNotePredictor** — classifies molecules as top/middle/base notes based on structural features

## Requirements

- Python 3.8+
- numpy

No rdkit or torch required.
