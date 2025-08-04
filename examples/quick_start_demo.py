#!/usr/bin/env python3
"""
Quick Start Demo for Smell Diffusion Generator

This demo shows the basic functionality of generating fragrance molecules
from text descriptions with safety evaluation.
"""

from smell_diffusion import SmellDiffusion, SafetyEvaluator


def main():
    """Run the quick start demo."""
    print("🌸🧪 Smell Diffusion Generator - Quick Start Demo")
    print("=" * 50)
    
    # Load pre-trained model
    print("\n1. Loading pre-trained model...")
    model = SmellDiffusion.from_pretrained('smell-diffusion-base-v1')
    safety = SafetyEvaluator()
    
    # Generate molecule from text description
    print("\n2. Generating molecules from text description...")
    prompt = "A fresh, aquatic fragrance with notes of sea breeze, cucumber, and white musk"
    print(f"Prompt: '{prompt}'")
    
    molecules = model.generate(
        prompt=prompt,
        num_molecules=5,
        guidance_scale=7.5,
        safety_filter=True
    )
    
    # Evaluate generated molecules
    print(f"\n3. Generated {len(molecules)} molecules:")
    print("-" * 40)
    
    for i, mol in enumerate(molecules):
        print(f"\nMolecule {i+1}:")
        print(f"  SMILES: {mol.smiles}")
        print(f"  Molecular Weight: {mol.molecular_weight:.1f} g/mol")
        print(f"  LogP: {mol.logp:.2f}")
        print(f"  Predicted notes: {mol.fragrance_notes.top + mol.fragrance_notes.middle + mol.fragrance_notes.base}")
        print(f"  Intensity: {mol.intensity:.1f}/10")
        print(f"  Longevity: {mol.longevity}")
        
        # Safety evaluation
        safety_report = safety.evaluate(mol)
        print(f"  Safety score: {safety_report.score:.1f}/100")
        print(f"  IFRA compliant: {'✓' if safety_report.ifra_compliant else '✗'}")
        
        if safety_report.allergens:
            print(f"  Potential allergens: {', '.join(safety_report.allergens)}")
        
        if safety_report.warnings:
            print(f"  Warnings: {'; '.join(safety_report.warnings)}")
    
    # Demonstrate different fragrance categories
    print(f"\n4. Testing different fragrance categories:")
    print("-" * 40)
    
    test_prompts = [
        "Elegant rose with woody undertones",
        "Bright citrus burst with bergamot and lemon",
        "Warm vanilla and amber base notes",
        "Fresh marine scent with ozonic accord"
    ]
    
    for prompt in test_prompts:
        mol = model.generate(prompt=prompt, num_molecules=1)
        if mol:
            print(f"\nPrompt: '{prompt}'")
            print(f"  Generated: {mol.smiles}")
            print(f"  Notes: {', '.join(mol.fragrance_notes.top + mol.fragrance_notes.middle + mol.fragrance_notes.base)}")
            print(f"  Safety: {mol.get_safety_profile().score:.0f}/100")
    
    # Model information
    print(f"\n5. Model Information:")
    print("-" * 40)
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("  - Try different text prompts")
    print("  - Explore multi-modal generation") 
    print("  - Use the comprehensive safety evaluator")
    print("  - Visualize molecules in 3D")


if __name__ == "__main__":
    main()