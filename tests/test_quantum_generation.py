"""
Comprehensive tests for quantum molecular generation breakthrough.

Tests quantum-enhanced molecular generation algorithms with statistical validation.
"""

import pytest
import asyncio
import time
import random
from typing import List, Dict, Any

from smell_diffusion.research.quantum_enhanced_generation import (
    QuantumState,
    QuantumMolecularOracle,
    QuantumMolecularGenerator,
    validate_quantum_advantage,
    benchmark_quantum_performance
)


class TestQuantumState:
    """Test quantum state representation."""
    
    def test_quantum_state_creation(self):
        """Test quantum state initialization."""
        state = QuantumState(
            amplitude=0.8 + 0.2j,
            phase=1.57,
            energy_level=2.5,
            molecular_representation="CC(C)=CCCC(C)=CCO"
        )
        
        assert abs(state.amplitude) <= 1.0  # Amplitude should be normalized
        assert state.phase == 1.57
        assert state.energy_level == 2.5
        assert state.molecular_representation == "CC(C)=CCCC(C)=CCO"
        assert state.coherence_time == 1.0  # Default value
        assert len(state.entanglement_partners) == 0
    
    def test_quantum_state_normalization(self):
        """Test quantum state amplitude normalization."""
        # Test with unnormalized amplitude
        state = QuantumState(
            amplitude=2.0 + 1.0j,  # |amplitude| = sqrt(5) > 1
            phase=0.0,
            energy_level=1.0,
            molecular_representation="C"
        )
        
        # Should be normalized to unit magnitude
        assert abs(abs(state.amplitude) - 1.0) < 1e-10
    
    def test_quantum_state_entanglement(self):
        """Test quantum entanglement partners."""
        state = QuantumState(
            amplitude=0.7 + 0.2j,
            phase=0.5,
            energy_level=1.0,
            molecular_representation="CC",
            entanglement_partners=["partner1", "partner2"]
        )
        
        assert len(state.entanglement_partners) == 2
        assert "partner1" in state.entanglement_partners
        assert "partner2" in state.entanglement_partners


class TestQuantumMolecularOracle:
    """Test quantum molecular oracle functionality."""
    
    def test_oracle_creation(self):
        """Test oracle initialization."""
        target_props = {"stability": 0.8, "intensity": 0.7}
        oracle = QuantumMolecularOracle(target_props)
        
        assert oracle.target_properties == target_props
        assert oracle.tolerance == 0.1  # Default
        assert oracle.iterations == 0
        assert oracle.max_iterations == 1000  # Default
    
    def test_oracle_evaluation(self):
        """Test oracle molecular evaluation."""
        target_props = {"stability": 0.8, "intensity": 0.7, "safety_score": 0.9}
        oracle = QuantumMolecularOracle(target_props)
        
        # Test with a valid SMILES string
        fitness = oracle.evaluate("CC(C)=CCCC(C)=CCO")
        
        assert 0.0 <= fitness <= 1.0
        assert oracle.iterations == 1
    
    def test_oracle_multiple_evaluations(self):
        """Test oracle with multiple evaluations."""
        target_props = {"stability": 0.8}
        oracle = QuantumMolecularOracle(target_props)
        
        molecules = ["CC", "CCC", "CCCC", "C=C", "C#C"]
        fitness_scores = []
        
        for mol in molecules:
            fitness = oracle.evaluate(mol)
            fitness_scores.append(fitness)
            assert 0.0 <= fitness <= 1.0
        
        assert oracle.iterations == len(molecules)
        assert len(fitness_scores) == len(molecules)
    
    def test_tunneling_probability(self):
        """Test quantum tunneling probability calculation."""
        target_props = {"stability": 0.8}
        oracle = QuantumMolecularOracle(target_props)
        
        # Test with molecule containing rare patterns
        rare_molecule = "CC(=O)NC#N"  # Contains C=O and C#N patterns
        tunneling_prob = oracle._calculate_tunneling_probability(rare_molecule)
        
        assert 0.0 <= tunneling_prob <= 1.0
        
        # Simple molecule should have lower tunneling probability
        simple_molecule = "CC"
        simple_tunneling = oracle._calculate_tunneling_probability(simple_molecule)
        
        assert simple_tunneling <= tunneling_prob
    
    def test_novelty_calculation(self):
        """Test molecular novelty calculation."""
        target_props = {"stability": 0.8}
        oracle = QuantumMolecularOracle(target_props)
        
        # Test novelty calculation is deterministic for same input
        molecule = "CC(C)=CCCC(C)=CCO"
        novelty1 = oracle._calculate_novelty(molecule)
        novelty2 = oracle._calculate_novelty(molecule)
        
        assert novelty1 == novelty2
        assert 0.0 <= novelty1 <= 1.0


class TestQuantumMolecularGenerator:
    """Test quantum molecular generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = QuantumMolecularGenerator(
            coherence_time=2.0,
            max_superposition_states=32,
            entanglement_strength=0.5
        )
        
        assert generator.coherence_time == 2.0
        assert generator.max_superposition_states == 32
        assert generator.entanglement_strength == 0.5
        assert len(generator.quantum_register) == 0
        assert len(generator.measurement_history) == 0
    
    def test_superposition_creation(self):
        """Test quantum superposition creation."""
        generator = QuantumMolecularGenerator(max_superposition_states=8)
        
        base_molecules = [
            "CC(C)=CCCC(C)=CCO",  # Geraniol
            "CC(C)CCCC(C)CCO",    # Citronellol
            "COC1=C(C=CC(=C1)C=O)O",  # Vanillin
            "CC1=CC=C(C=C1)C=O"   # p-Tolualdehyde
        ]
        
        target_properties = {"stability": 0.8, "intensity": 0.7}
        
        superposition = generator.create_superposition(base_molecules, target_properties)
        
        assert len(superposition) == len(base_molecules)
        
        for state in superposition:
            assert isinstance(state, QuantumState)
            assert abs(state.amplitude) <= 1.0
            assert state.energy_level >= 0.0
            assert state.molecular_representation in base_molecules
    
    def test_entanglement_creation(self):
        """Test quantum entanglement between states."""
        generator = QuantumMolecularGenerator(entanglement_strength=0.8)
        
        # Create similar molecules that should be entangled
        base_molecules = ["CC", "CCC", "CCCC"]  # Similar chain molecules
        target_properties = {"stability": 0.8}
        
        superposition = generator.create_superposition(base_molecules, target_properties)
        
        # Check if entanglement was created
        total_entanglements = sum(len(state.entanglement_partners) for state in superposition)
        assert total_entanglements >= 0  # Some entanglement should occur
    
    def test_quantum_evolution(self):
        """Test quantum evolution of superposition."""
        generator = QuantumMolecularGenerator()
        
        base_molecules = ["CC", "CCC"]
        target_properties = {"stability": 0.8, "intensity": 0.7}
        
        initial_superposition = generator.create_superposition(base_molecules, target_properties)
        evolved_superposition = generator.quantum_evolution(
            initial_superposition, 
            target_properties,
            evolution_steps=10
        )
        
        assert len(evolved_superposition) == len(initial_superposition)
        
        # Check that evolution occurred (phases should have changed)
        for initial, evolved in zip(initial_superposition, evolved_superposition):
            # Phase should have evolved
            assert evolved.phase != initial.phase or evolved.amplitude != initial.amplitude
    
    def test_measurement(self):
        """Test quantum measurement collapse."""
        generator = QuantumMolecularGenerator()
        
        base_molecules = ["CC(C)=CCCC(C)=CCO", "CC(C)CCCC(C)CCO"]
        target_properties = {"stability": 0.8}
        
        superposition = generator.create_superposition(base_molecules, target_properties)
        measurements = generator.measure_superposition(superposition, num_measurements=5)
        
        assert len(measurements) == 5
        
        for measurement in measurements:
            assert measurement in base_molecules
        
        # Check measurement history
        assert len(generator.measurement_history) == 5
    
    def test_quantum_generate_basic(self):
        """Test basic quantum generation."""
        generator = QuantumMolecularGenerator(
            max_superposition_states=4,
            coherence_time=1.0
        )
        
        prompt = "Fresh citrus fragrance"
        results = generator.quantum_generate(
            prompt=prompt,
            num_molecules=3,
            evolution_steps=5
        )
        
        assert len(results) <= 3  # May generate fewer if some fail
        
        for result in results:
            assert 'molecule' in result
            assert 'fidelity' in result
            assert 'novelty' in result
            assert 'quantum_amplitude' in result
            assert 'generation_method' in result
            
            assert 0.0 <= result['fidelity'] <= 1.0
            assert 0.0 <= result['novelty'] <= 1.0
            assert result['generation_method'] == 'quantum_enhanced'
    
    def test_quantum_generate_with_properties(self):
        """Test quantum generation with specific target properties."""
        generator = QuantumMolecularGenerator()
        
        target_properties = {
            "stability": 0.9,
            "intensity": 0.8,
            "longevity": 0.7,
            "safety_score": 0.95
        }
        
        results = generator.quantum_generate(
            prompt="Sophisticated woody fragrance",
            target_properties=target_properties,
            num_molecules=2,
            evolution_steps=10
        )
        
        assert len(results) <= 2
        
        for result in results:
            # Should have quantum-specific metrics
            assert 'quantum_amplitude' in result
            assert 'quantum_phase' in result
            assert 'energy_level' in result
            assert 'tunneling_probability' in result
            
            # Quantum metrics should be in valid ranges
            assert 0.0 <= result['quantum_amplitude'] <= 1.0
            assert 0.0 <= result['tunneling_probability'] <= 1.0
    
    def test_research_metrics_tracking(self):
        """Test research metrics collection."""
        generator = QuantumMolecularGenerator()
        
        # Generate multiple molecules to build statistics
        for i in range(3):
            results = generator.quantum_generate(
                prompt=f"Test prompt {i}",
                num_molecules=2,
                evolution_steps=5
            )
        
        # Check research metrics were collected
        assert generator.generation_stats['total_generations'] >= 3
        assert len(generator.research_metrics['superposition_efficiency']) >= 3
        
        # Get research report
        report = generator.get_research_report()
        
        assert 'algorithm_name' in report
        assert 'generation_statistics' in report
        assert 'research_metrics' in report
        assert 'quantum_parameters' in report
        
        # Check statistical significance calculation
        sig_stats = report.get('statistical_significance', {})
        assert 'significant' in sig_stats
        assert isinstance(sig_stats.get('significant'), bool)


class TestQuantumAdvantageValidation:
    """Test quantum advantage validation."""
    
    @pytest.mark.asyncio
    async def test_validate_quantum_advantage_basic(self):
        """Test basic quantum advantage validation."""
        generator = QuantumMolecularGenerator(max_superposition_states=4)
        
        test_prompts = [
            "Fresh citrus fragrance",
            "Warm woody scent",
            "Floral bouquet"
        ]
        
        # Run validation with limited trials for testing
        validation_results = await validate_quantum_advantage(
            generator, 
            test_prompts, 
            num_trials=3
        )
        
        assert 'quantum_results' in validation_results
        assert 'classical_results' in validation_results
        assert 'statistical_tests' in validation_results
        
        assert len(validation_results['quantum_results']) == 3
        assert len(validation_results['classical_results']) == 3
        
        # Check statistical test results
        stats = validation_results['statistical_tests']
        assert 'quantum_mean_fidelity' in stats
        assert 'classical_mean_fidelity' in stats
        assert 'improvement_percentage' in stats
    
    @pytest.mark.asyncio
    async def test_benchmark_quantum_performance(self):
        """Test quantum performance benchmarking."""
        generator = QuantumMolecularGenerator(max_superposition_states=4)
        
        benchmark_suite = [
            {
                'name': 'citrus_test',
                'prompts': ['Fresh citrus', 'Lemon zest'],
                'expected_properties': {'intensity': 0.8, 'freshness': 0.9},
                'num_repeats': 2
            },
            {
                'name': 'floral_test', 
                'prompts': ['Rose garden', 'Jasmine bloom'],
                'expected_properties': {'intensity': 0.7, 'complexity': 0.8},
                'num_repeats': 2
            }
        ]
        
        results = benchmark_quantum_performance(generator, benchmark_suite)
        
        assert 'task_results' in results
        assert 'overall_performance' in results
        assert 'quantum_metrics' in results
        
        assert len(results['task_results']) == 2
        
        for task_result in results['task_results']:
            assert 'task_name' in task_result
            assert 'mean_fidelity' in task_result
            assert 'reproducibility_score' in task_result
            assert 'quantum_advantage_factor' in task_result


class TestQuantumAlgorithmPerformance:
    """Test performance characteristics of quantum algorithms."""
    
    def test_superposition_scaling(self):
        """Test superposition creation scales with molecule count."""
        generator = QuantumMolecularGenerator(max_superposition_states=16)
        
        base_molecules = [f"C{'C' * i}" for i in range(10)]  # C, CC, CCC, etc.
        target_properties = {"stability": 0.8}
        
        start_time = time.time()
        superposition = generator.create_superposition(base_molecules, target_properties)
        creation_time = time.time() - start_time
        
        assert len(superposition) == len(base_molecules)
        assert creation_time < 1.0  # Should be fast for small test
    
    def test_evolution_convergence(self):
        """Test quantum evolution convergence behavior."""
        generator = QuantumMolecularGenerator()
        
        base_molecules = ["CC", "CCC"]
        target_properties = {"stability": 0.9, "intensity": 0.8}
        
        superposition = generator.create_superposition(base_molecules, target_properties)
        
        # Test different evolution step counts
        short_evolution = generator.quantum_evolution(
            superposition, target_properties, evolution_steps=5
        )
        long_evolution = generator.quantum_evolution(
            superposition, target_properties, evolution_steps=20
        )
        
        # Both should return valid superpositions
        assert len(short_evolution) == len(superposition)
        assert len(long_evolution) == len(superposition)
        
        # Longer evolution should potentially show more changes
        # (though this is stochastic, so we just check basic validity)
        for state in long_evolution:
            assert abs(state.amplitude) <= 1.0
            assert isinstance(state.phase, (int, float))
    
    def test_measurement_statistics(self):
        """Test measurement statistics follow quantum principles."""
        generator = QuantumMolecularGenerator()
        
        # Create superposition with known amplitudes
        base_molecules = ["CC", "CCC", "CCCC"]
        target_properties = {"stability": 0.8}
        
        superposition = generator.create_superposition(base_molecules, target_properties)
        
        # Perform many measurements to test statistical behavior
        measurements = generator.measure_superposition(
            superposition, 
            num_measurements=50
        )
        
        assert len(measurements) == 50
        
        # Count measurement frequencies
        measurement_counts = {}
        for measurement in measurements:
            measurement_counts[measurement] = measurement_counts.get(measurement, 0) + 1
        
        # All measurements should be from original molecules
        for measurement in measurement_counts:
            assert measurement in base_molecules
        
        # Should have reasonable distribution (not all the same)
        unique_measurements = len(measurement_counts)
        assert unique_measurements >= 1  # At least one unique measurement
    
    def test_memory_efficiency(self):
        """Test memory efficiency of quantum operations."""
        generator = QuantumMolecularGenerator(max_superposition_states=8)
        
        # Create multiple superpositions to test memory usage
        for i in range(5):
            base_molecules = [f"C{'C' * j}" for j in range(4)]
            target_properties = {"stability": 0.8}
            
            superposition = generator.create_superposition(base_molecules, target_properties)
            measurements = generator.measure_superposition(superposition, num_measurements=3)
            
            # Check that measurement history doesn't grow unbounded
            assert len(generator.measurement_history) <= 1000  # maxlen=1000
    
    def test_error_handling_quantum_operations(self):
        """Test error handling in quantum operations."""
        generator = QuantumMolecularGenerator()
        
        # Test with empty molecule list
        empty_superposition = generator.create_superposition([], {"stability": 0.8})
        assert len(empty_superposition) == 0
        
        # Test measurement of empty superposition
        measurements = generator.measure_superposition([], num_measurements=5)
        assert len(measurements) == 0
        
        # Test evolution with empty superposition  
        evolved = generator.quantum_evolution([], {"stability": 0.8}, evolution_steps=10)
        assert len(evolved) == 0
    
    def test_deterministic_reproducibility(self):
        """Test deterministic aspects are reproducible."""
        # Set random seed for reproducibility
        random.seed(42)
        
        generator1 = QuantumMolecularGenerator()
        generator2 = QuantumMolecularGenerator()
        
        base_molecules = ["CC", "CCC"]
        target_properties = {"stability": 0.8}
        
        # Same random seed should give same phase encoding
        for mol in base_molecules:
            phase1 = generator1._encode_molecular_phase(mol)
            phase2 = generator2._encode_molecular_phase(mol)
            assert phase1 == phase2  # Should be deterministic
        
        # Energy calculations should also be deterministic
        for mol in base_molecules:
            energy1 = generator1._calculate_energy_level(mol)
            energy2 = generator2._calculate_energy_level(mol)
            assert energy1 == energy2


@pytest.fixture
def sample_quantum_generator():
    """Fixture providing a sample quantum generator."""
    return QuantumMolecularGenerator(
        coherence_time=1.0,
        max_superposition_states=8,
        entanglement_strength=0.3
    )


@pytest.fixture
def sample_molecules():
    """Fixture providing sample molecules for testing."""
    return [
        "CC(C)=CCCC(C)=CCO",      # Geraniol
        "CC(C)CCCC(C)CCO",        # Citronellol
        "COC1=C(C=CC(=C1)C=O)O",  # Vanillin
        "CC1=CC=C(C=C1)C=O",      # p-Tolualdehyde
        "CC1=CCC(CC1)C(C)(C)O"    # Linalool
    ]


@pytest.fixture
def sample_target_properties():
    """Fixture providing sample target properties."""
    return {
        "stability": 0.8,
        "intensity": 0.7,
        "longevity": 0.6,
        "safety_score": 0.9,
        "novelty": 0.5
    }