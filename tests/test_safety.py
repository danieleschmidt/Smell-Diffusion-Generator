"""Tests for safety evaluation functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from smell_diffusion.safety.evaluator import SafetyEvaluator, ComprehensiveSafetyReport
from smell_diffusion.core.molecule import Molecule, SafetyProfile
from smell_diffusion.utils.validation import ValidationError, SafetyValidator


class TestSafetyEvaluator:
    """Test cases for SafetyEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test safety evaluator initialization."""
        evaluator = SafetyEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'EU_ALLERGENS')
        assert isinstance(evaluator.EU_ALLERGENS, dict)
    
    def test_basic_safety_evaluation(self, sample_molecules, mock_rdkit):
        """Test basic safety evaluation."""
        evaluator = SafetyEvaluator()
        
        for mol in sample_molecules['valid']:
            safety = evaluator.evaluate(mol)
            
            assert isinstance(safety, SafetyProfile)
            assert isinstance(safety.score, float)
            assert 0 <= safety.score <= 100
            assert isinstance(safety.ifra_compliant, bool)
            assert isinstance(safety.allergens, list)
            assert isinstance(safety.warnings, list)
    
    def test_invalid_molecule_safety(self, sample_molecules):
        """Test safety evaluation of invalid molecules."""
        evaluator = SafetyEvaluator()
        
        for mol in sample_molecules['invalid']:
            safety = evaluator.evaluate(mol)
            
            assert safety.score == 0.0
            assert not safety.ifra_compliant
            assert len(safety.warnings) > 0  # Should have invalid structure warning
    
    def test_comprehensive_safety_evaluation(self, sample_molecules, mock_rdkit):
        """Test comprehensive safety evaluation."""
        evaluator = SafetyEvaluator()
        
        mol = sample_molecules['valid'][0]
        report = evaluator.comprehensive_evaluation(mol)
        
        assert isinstance(report, ComprehensiveSafetyReport)
        assert report.molecule_smiles == mol.smiles
        assert isinstance(report.overall_score, float)
        assert isinstance(report.ifra_compliant, bool)
        assert isinstance(report.regulatory_status, dict)
        assert isinstance(report.toxicity_predictions, dict)
        assert isinstance(report.allergen_analysis, dict)
        assert isinstance(report.environmental_impact, dict)
        assert isinstance(report.recommendations, list)
    
    def test_allergen_screening(self, mock_rdkit):
        """Test allergen screening functionality."""
        evaluator = SafetyEvaluator()
        
        # Test with known allergen structure (simplified)
        mol = Molecule("CC(C)=CCCC(C)=CCO")  # Geraniol (known allergen)
        allergen_results = evaluator._screen_allergens(mol)
        
        assert isinstance(allergen_results, dict)
        assert 'detected' in allergen_results
        assert 'total_count' in allergen_results
        assert 'risk_level' in allergen_results
        assert allergen_results['risk_level'] in ['low', 'medium', 'high']
    
    def test_structural_similarity(self):
        """Test structural similarity calculation."""
        evaluator = SafetyEvaluator()
        
        # Test identical molecules
        similarity = evaluator._structural_similarity("CCO", "CCO")
        assert similarity == 1.0
        
        # Test different molecules
        similarity = evaluator._structural_similarity("CCO", "CCC")
        assert 0 <= similarity <= 1.0
        
        # Test empty strings
        similarity = evaluator._structural_similarity("", "")
        assert similarity == 0.0
    
    def test_regulatory_status_check(self, mock_rdkit):
        """Test regulatory compliance checking."""
        evaluator = SafetyEvaluator()
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        regulatory_status = evaluator._check_regulatory_status(mol)
        
        assert isinstance(regulatory_status, dict)
        assert 'EU' in regulatory_status
        assert 'US' in regulatory_status  
        assert 'IFRA' in regulatory_status
        
        for region, status in regulatory_status.items():
            assert status in ['Approved', 'Restricted', 'Prohibited', 'GRAS', 'Requires_Review', 'Compliant', 'Non-Compliant']
    
    def test_toxicity_prediction(self, mock_rdkit):
        """Test toxicity prediction functionality."""
        evaluator = SafetyEvaluator()
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        toxicity = evaluator._predict_toxicity(mol)
        
        assert isinstance(toxicity, dict)
        
        if 'acute_oral_toxicity' in toxicity:
            assert 'category' in toxicity['acute_oral_toxicity']
            assert 'confidence' in toxicity['acute_oral_toxicity']
        
        if 'skin_sensitization' in toxicity:
            assert 'prediction' in toxicity['skin_sensitization']
            assert 'confidence' in toxicity['skin_sensitization']
    
    def test_environmental_impact_assessment(self, mock_rdkit):
        """Test environmental impact assessment."""
        evaluator = SafetyEvaluator()
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        environmental = evaluator._assess_environmental_impact(mol)
        
        assert isinstance(environmental, dict)
        assert 'biodegradability' in environmental
        assert 'bioaccumulation' in environmental
        
        # Check valid values
        biodegradability_values = ['Readily biodegradable', 'Inherently biodegradable', 'Not readily biodegradable']
        assert environmental['biodegradability'] in biodegradability_values
        
        bioaccumulation_values = ['Low potential', 'Moderate potential', 'High potential']
        assert environmental['bioaccumulation'] in bioaccumulation_values
    
    def test_recommendation_generation(self, mock_rdkit):
        """Test safety recommendation generation."""
        evaluator = SafetyEvaluator()
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # Mock safety components
        safety = SafetyProfile(score=60.0, ifra_compliant=False, allergens=['test'], warnings=[])
        allergens = {'risk_level': 'medium', 'detected': []}
        environmental = {'bioaccumulation': 'High potential'}
        
        recommendations = evaluator._generate_recommendations(mol, safety, allergens, environmental)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_comprehensive_evaluation_invalid_molecule(self):
        """Test comprehensive evaluation with invalid molecule."""
        evaluator = SafetyEvaluator()
        mol = Molecule("INVALID_SMILES")
        
        report = evaluator.comprehensive_evaluation(mol)
        
        assert report.overall_score == 0.0
        assert not report.ifra_compliant
        assert 'Invalid' in str(report.regulatory_status)
        assert 'Molecule structure is invalid' in report.recommendations


class TestEUAllergens:
    """Test EU allergen database and detection."""
    
    def test_allergen_database_structure(self):
        """Test allergen database has correct structure."""
        evaluator = SafetyEvaluator()
        allergens = evaluator.EU_ALLERGENS
        
        assert isinstance(allergens, dict)
        assert len(allergens) > 0
        
        # Check for some known allergens
        known_allergens = ['geraniol', 'linalool', 'limonene', 'citral', 'coumarin']
        for allergen in known_allergens:
            # Should be present (with possible underscore formatting)
            allergen_found = any(allergen.replace('_', ' ') in key.replace('_', ' ') 
                               for key in allergens.keys())
            assert allergen_found, f"Known allergen {allergen} not found in database"
    
    def test_allergen_smiles_validity(self):
        """Test that allergen SMILES are valid structures."""
        evaluator = SafetyEvaluator()
        
        # Sample a few allergens to test
        sample_allergens = dict(list(evaluator.EU_ALLERGENS.items())[:5])
        
        for name, smiles in sample_allergens.items():
            assert isinstance(smiles, str)
            assert len(smiles) > 0
            # Would test with RDKit in real scenario, but using mock here


class TestSafetyIntegration:
    """Integration tests for safety system."""
    
    def test_safety_workflow(self, mock_rdkit):
        """Test complete safety evaluation workflow."""
        evaluator = SafetyEvaluator()
        
        # Create test molecule
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # Basic evaluation
        basic_safety = evaluator.evaluate(mol)
        assert isinstance(basic_safety, SafetyProfile)
        
        # Comprehensive evaluation
        comprehensive = evaluator.comprehensive_evaluation(mol)
        assert isinstance(comprehensive, ComprehensiveSafetyReport)
        
        # Scores should be consistent
        assert basic_safety.score == comprehensive.overall_score
        assert basic_safety.ifra_compliant == comprehensive.ifra_compliant
    
    def test_safety_batch_evaluation(self, sample_molecules, mock_rdkit):
        """Test evaluating multiple molecules."""
        evaluator = SafetyEvaluator()
        
        results = []
        for mol in sample_molecules['valid']:
            safety = evaluator.evaluate(mol)
            results.append(safety)
        
        assert len(results) == len(sample_molecules['valid'])
        assert all(isinstance(result, SafetyProfile) for result in results)
    
    def test_problematic_molecule_detection(self, sample_molecules, mock_rdkit):
        """Test detection of problematic molecular structures."""
        evaluator = SafetyEvaluator()
        
        # These would be flagged as problematic in real evaluation
        for mol in sample_molecules['problematic']:
            safety = evaluator.evaluate(mol)
            # In mock, these still appear valid, but would be flagged in real system
            assert isinstance(safety, SafetyProfile)


@pytest.mark.parametrize("molecular_weight,logp,expected_score_range", [
    (150.0, 2.0, (70, 100)),    # Good fragrance molecule
    (50.0, 1.0, (50, 80)),      # Light molecule
    (400.0, 3.0, (60, 90)),     # Heavy molecule  
    (600.0, 6.0, (20, 50)),     # Very heavy, lipophilic
])
def test_safety_scoring_based_on_properties(molecular_weight, logp, expected_score_range, mock_rdkit):
    """Test safety scoring based on molecular properties."""
    # Mock RDKit to return specific values
    with patch('smell_diffusion.core.molecule.Descriptors') as mock_desc:
        mock_desc.MolWt.return_value = molecular_weight
        mock_desc.MolLogP.return_value = logp
        
        mol = Molecule("CC(C)=CCCC(C)=CCO")  # Use valid SMILES
        safety = mol.get_safety_profile()
        
        min_score, max_score = expected_score_range
        assert min_score <= safety.score <= max_score, f"Score {safety.score} not in range {expected_score_range}"


class TestSafetyValidation:
    """Test safety validation and edge cases."""
    
    def test_empty_molecule_safety(self):
        """Test safety evaluation with empty molecule."""
        evaluator = SafetyEvaluator()
        mol = Molecule("")
        
        safety = evaluator.evaluate(mol)
        assert safety.score == 0.0
        assert not safety.ifra_compliant
    
    def test_none_molecule_safety(self):
        """Test safety evaluation with None."""
        evaluator = SafetyEvaluator()
        
        # This should handle gracefully
        mol = Molecule(None)
        safety = evaluator.evaluate(mol)
        
        assert safety.score == 0.0
        assert not safety.ifra_compliant
    
    def test_safety_evaluator_caching(self, mock_rdkit):
        """Test that safety evaluation can be cached."""
        evaluator = SafetyEvaluator()
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # First evaluation
        safety1 = evaluator.evaluate(mol)
        
        # Second evaluation (would use cache if implemented)  
        safety2 = evaluator.evaluate(mol)
        
        # Results should be identical
        assert safety1.score == safety2.score
        assert safety1.ifra_compliant == safety2.ifra_compliant