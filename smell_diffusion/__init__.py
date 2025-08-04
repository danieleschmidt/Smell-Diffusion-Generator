"""
Smell Diffusion Generator

Cross-modal diffusion model that generates odorant molecules from text descriptions,
with integrated safety evaluation and fragrance note prediction.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .core.smell_diffusion import SmellDiffusion
from .core.molecule import Molecule
from .safety.evaluator import SafetyEvaluator
from .multimodal.generator import MultiModalGenerator
from .editing.editor import MoleculeEditor
from .design.accord import AccordDesigner

__all__ = [
    "SmellDiffusion",
    "Molecule", 
    "SafetyEvaluator",
    "MultiModalGenerator",
    "MoleculeEditor",
    "AccordDesigner",
]