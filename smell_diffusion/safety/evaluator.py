"""Safety evaluation for generated fragrance molecules."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.molecule import Molecule, SafetyProfile


@dataclass
class ComprehensiveSafetyReport:
    """Detailed safety evaluation report."""
    molecule_smiles: str
    overall_score: float
    ifra_compliant: bool
    regulatory_status: Dict[str, str]
    toxicity_predictions: Dict[str, Any]
    allergen_analysis: Dict[str, Any]
    environmental_impact: Dict[str, Any]
    recommendations: List[str]


class SafetyEvaluator:
    """Comprehensive safety evaluation for fragrance molecules."""
    
    # Known allergens from EU Regulation 1223/2009
    EU_ALLERGENS = {
        "amyl_cinnamal": "CC(C)CCOC(=O)/C=C/C1=CC=CC=C1",
        "amylcinnamyl_alcohol": "CC(C)CCOC/C=C/C1=CC=CC=C1", 
        "anise_alcohol": "COC1=CC=C(C=C1)CCO",
        "benzyl_alcohol": "C1=CC=C(C=C1)CO",
        "benzyl_benzoate": "C1=CC=C(C=C1)COC(=O)C2=CC=CC=C2",
        "benzyl_cinnamate": "C1=CC=C(C=C1)COC(=O)/C=C/C2=CC=CC=C2",
        "benzyl_salicylate": "C1=CC=C(C=C1)COC(=O)C2=CC=CC=C2O",
        "cinnamal": "C1=CC=C(C=C1)/C=C/C=O",
        "cinnamyl_alcohol": "C1=CC=C(C=C1)/C=C/CO",
        "citral": "CC(=CCCC(=CCO)C)C",
        "citronellol": "CC(CCCC(C)=CCO)C",
        "coumarin": "C1=CC=C2C(=C1)C=CC(=O)O2",
        "eugenol": "COC1=C(C=CC(=C1)CC=C)O",
        "farnesol": "CC(=CCCC(=CCCC(=CCO)C)C)C",
        "geraniol": "CC(=CCCC(=CCO)C)C",
        "hexyl_cinnamal": "CCCCCCOC(=O)/C=C/C1=CC=CC=C1",
        "hydroxycitronellal": "CC(CCC(CCO)(C)O)C",
        "hydroxyisohexyl_cinnamaldehyde": "CC(C)CC(C)C(=O)/C=C/C1=CC=CC=C1",
        "isoeugenol": "COC1=C(C=CC(=C1)/C=C/C)O",
        "limonene": "CC1=CCC(CC1)C(=C)C",
        "linalool": "CC(C)(C=C)CCC(=CCO)C",
        "methyl_heptine_carbonate": "C#CCCCCCOC(=O)OC",
        "oak_moss": "COC1=CC(=CC=C1O)C(=O)/C=C/C2=CC=CC=C2",
        "tree_moss": "COC1=CC(=CC=C1O)C(=O)C2=CC=CC=C2"
    }
    
    def __init__(self):
        """Initialize safety evaluator."""
        pass
    
    def evaluate(self, molecule: Molecule) -> SafetyProfile:
        """Perform basic safety evaluation."""
        return molecule.get_safety_profile()
    
    def comprehensive_evaluation(self, molecule: Molecule) -> ComprehensiveSafetyReport:
        """Perform comprehensive safety evaluation."""
        if not molecule.is_valid:
            return ComprehensiveSafetyReport(
                molecule_smiles=molecule.smiles,
                overall_score=0.0,
                ifra_compliant=False,
                regulatory_status={"EU": "Invalid", "US": "Invalid"},
                toxicity_predictions={},
                allergen_analysis={"detected": [], "risk_level": "unknown"},
                environmental_impact={"biodegradability": "unknown"},
                recommendations=["Molecule structure is invalid"]
            )
        
        # Basic safety profile
        basic_safety = self.evaluate(molecule)
        
        # Allergen screening
        allergen_results = self._screen_allergens(molecule)
        
        # Regulatory check
        regulatory_status = self._check_regulatory_status(molecule)
        
        # Environmental assessment
        environmental = self._assess_environmental_impact(molecule)
        
        # Toxicity predictions
        toxicity = self._predict_toxicity(molecule)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            molecule, basic_safety, allergen_results, environmental
        )
        
        return ComprehensiveSafetyReport(
            molecule_smiles=molecule.smiles,
            overall_score=basic_safety.score,
            ifra_compliant=basic_safety.ifra_compliant,
            regulatory_status=regulatory_status,
            toxicity_predictions=toxicity,
            allergen_analysis=allergen_results,
            environmental_impact=environmental,
            recommendations=recommendations
        )
    
    def _screen_allergens(self, molecule: Molecule) -> Dict[str, Any]:
        """Screen for known allergens."""
        detected_allergens = []
        
        # Check against known allergen patterns
        mol_smiles = molecule.smiles
        
        for allergen_name, allergen_smiles in self.EU_ALLERGENS.items():
            # Simple substring matching (in practice, would use proper structural matching)
            if self._structural_similarity(mol_smiles, allergen_smiles) > 0.8:
                detected_allergens.append({
                    "name": allergen_name.replace("_", " ").title(),
                    "similarity": self._structural_similarity(mol_smiles, allergen_smiles),
                    "regulation": "EU Cosmetics Regulation"
                })
        
        # Determine risk level
        if len(detected_allergens) > 2:
            risk_level = "high"
        elif len(detected_allergens) > 0:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "detected": detected_allergens,
            "total_count": len(detected_allergens),
            "risk_level": risk_level
        }
    
    def _structural_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate structural similarity between two SMILES."""
        # Simplified similarity calculation
        if smiles1 == smiles2:
            return 1.0
        
        # Check for common substructures
        common_patterns = 0
        total_patterns = 0
        
        # Simple pattern matching (in practice would use molecular fingerprints)
        patterns = ["C=C", "C=O", "C1=CC=CC=C1", "CC(C)", "CO", "C#C"]
        
        for pattern in patterns:
            total_patterns += 1
            if pattern in smiles1 and pattern in smiles2:
                common_patterns += 1
        
        return common_patterns / total_patterns if total_patterns > 0 else 0.0
    
    def _check_regulatory_status(self, molecule: Molecule) -> Dict[str, str]:
        """Check regulatory compliance status."""
        basic_safety = molecule.get_safety_profile()
        
        # EU status
        eu_status = "Approved" if basic_safety.ifra_compliant else "Restricted"
        if basic_safety.score < 30:
            eu_status = "Prohibited"
        
        # US FDA status (simplified)
        us_status = "GRAS" if basic_safety.score > 80 else "Requires_Review"
        if basic_safety.score < 30:
            us_status = "Prohibited"
        
        return {
            "EU": eu_status,
            "US": us_status,
            "IFRA": "Compliant" if basic_safety.ifra_compliant else "Non-Compliant"
        }
    
    def _predict_toxicity(self, molecule: Molecule) -> Dict[str, Any]:
        """Predict toxicity endpoints."""
        mw = molecule.molecular_weight
        logp = molecule.logp
        
        # Simple QSAR-like predictions
        predictions = {}
        
        # Acute oral toxicity (LD50)
        if mw < 100:
            ld50_category = "Category 3"  # Moderate toxicity
        elif mw < 300:
            ld50_category = "Category 4"  # Low toxicity
        else:
            ld50_category = "Category 5"  # Very low toxicity
        
        predictions["acute_oral_toxicity"] = {
            "category": ld50_category,
            "confidence": "low"
        }
        
        # Skin sensitization
        if logp > 4 and mw > 200:
            skin_sens = "Positive"
        elif logp < 1:
            skin_sens = "Negative" 
        else:
            skin_sens = "Inconclusive"
        
        predictions["skin_sensitization"] = {
            "prediction": skin_sens,
            "confidence": "medium"
        }
        
        return predictions
    
    def _assess_environmental_impact(self, molecule: Molecule) -> Dict[str, Any]:
        """Assess environmental impact."""
        mw = molecule.molecular_weight
        logp = molecule.logp
        
        # Biodegradability prediction
        if mw < 300 and logp < 4:
            biodegradability = "Readily biodegradable"
        elif mw < 500:
            biodegradability = "Inherently biodegradable"
        else:
            biodegradability = "Not readily biodegradable"
        
        # Bioaccumulation potential
        if logp > 4:
            bioaccumulation = "High potential"
        elif logp > 3:
            bioaccumulation = "Moderate potential"
        else:
            bioaccumulation = "Low potential"
        
        return {
            "biodegradability": biodegradability,
            "bioaccumulation": bioaccumulation,
            "aquatic_toxicity": "Low concern" if logp < 3 else "Moderate concern"
        }
    
    def _generate_recommendations(self, molecule: Molecule, safety: SafetyProfile,
                                allergens: Dict[str, Any], environmental: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if safety.score < 70:
            recommendations.append("Consider structural modifications to improve safety profile")
        
        if allergens["risk_level"] == "high":
            recommendations.append("High allergen risk - consider alternative structures")
        elif allergens["risk_level"] == "medium":
            recommendations.append("Moderate allergen risk - conduct patch testing")
        
        if not safety.ifra_compliant:
            recommendations.append("IFRA non-compliant - restrict usage concentrations")
        
        if environmental["bioaccumulation"] == "High potential":
            recommendations.append("High bioaccumulation potential - consider environmental impact")
        
        if molecule.molecular_weight > 400:
            recommendations.append("High molecular weight may affect skin penetration")
        
        if molecule.logp > 5:
            recommendations.append("High lipophilicity may cause skin irritation")
        
        if not recommendations:
            recommendations.append("Molecule shows good safety profile for fragrance use")
        
        return recommendations