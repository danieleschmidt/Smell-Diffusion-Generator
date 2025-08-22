"""
Comprehensive tests for advanced contrastive multimodal learning.

Tests contrastive learning algorithms for molecular-text alignment.
"""

import pytest
import asyncio
import random
import math
from typing import List, Dict, Any

from smell_diffusion.research.advanced_contrastive_learning import (
    ModalityType,
    ModalityEmbedding,
    ContrastiveExample,
    MultimodalEncoder,
    TextEncoder,
    MoleculeEncoder,
    OlfactoryEncoder,
    ProjectionHead,
    validate_contrastive_learning
)


class TestModalityEmbedding:
    """Test modality embedding representation."""
    
    def test_embedding_creation(self):
        """Test modality embedding initialization."""
        embedding = ModalityEmbedding(
            modality_type=ModalityType.TEXT,
            embedding=[0.5, 0.3, 0.8, 0.1],
            metadata={"source": "test"},
            confidence=0.9
        )
        
        assert embedding.modality_type == ModalityType.TEXT
        assert len(embedding.embedding) == 4
        assert embedding.metadata["source"] == "test"
        assert embedding.confidence == 0.9
    
    def test_embedding_normalization(self):
        """Test automatic embedding normalization."""
        # Create unnormalized embedding
        unnormalized = [3.0, 4.0, 0.0, 0.0]  # Magnitude = 5
        
        embedding = ModalityEmbedding(
            modality_type=ModalityType.MOLECULE,
            embedding=unnormalized
        )
        
        # Should be normalized to unit length
        magnitude = math.sqrt(sum(x*x for x in embedding.embedding))
        assert abs(magnitude - 1.0) < 1e-10
        
        # Direction should be preserved
        expected_normalized = [0.6, 0.8, 0.0, 0.0]
        for actual, expected in zip(embedding.embedding, expected_normalized):
            assert abs(actual - expected) < 1e-10
    
    def test_zero_embedding_handling(self):
        """Test handling of zero embeddings."""
        embedding = ModalityEmbedding(
            modality_type=ModalityType.OLFACTORY,
            embedding=[0.0, 0.0, 0.0]
        )
        
        # Zero embedding should remain zero
        assert all(x == 0.0 for x in embedding.embedding)


class TestContrastiveExample:
    """Test contrastive example structure."""
    
    def test_contrastive_example_creation(self):
        """Test contrastive example initialization."""
        anchor = ModalityEmbedding(ModalityType.TEXT, [0.5, 0.5, 0.0, 0.7])
        positive = ModalityEmbedding(ModalityType.MOLECULE, [0.6, 0.4, 0.1, 0.6])
        negative = ModalityEmbedding(ModalityType.MOLECULE, [0.1, 0.2, 0.9, 0.4])
        
        example = ContrastiveExample(
            anchor=anchor,
            positive_pairs=[(anchor, positive)],
            negative_pairs=[(anchor, negative)]
        )
        
        assert example.anchor == anchor
        assert len(example.positive_pairs) == 1
        assert len(example.negative_pairs) == 1
        assert len(example.example_id) == 8  # 8 character hash
    
    def test_get_all_embeddings(self):
        """Test retrieval of all embeddings in example."""
        anchor = ModalityEmbedding(ModalityType.TEXT, [1.0, 0.0])
        pos1 = ModalityEmbedding(ModalityType.MOLECULE, [0.0, 1.0])
        pos2 = ModalityEmbedding(ModalityType.OLFACTORY, [0.7, 0.7])
        neg1 = ModalityEmbedding(ModalityType.MOLECULE, [0.3, 0.9])
        neg2 = ModalityEmbedding(ModalityType.OLFACTORY, [0.2, 0.9])
        
        example = ContrastiveExample(
            anchor=anchor,
            positive_pairs=[(anchor, pos1), (pos2, anchor)],
            negative_pairs=[(anchor, neg1), (neg2, anchor)]
        )
        
        all_embeddings = example.get_all_embeddings()
        assert len(all_embeddings) == 9  # anchor + 2*2 + 2*2 = 9
        assert anchor in all_embeddings


class TestModalityEncoders:
    """Test individual modality encoders."""
    
    def test_text_encoder(self):
        """Test text encoder functionality."""
        encoder = TextEncoder(embedding_dim=128)
        
        # Test basic encoding
        text = "Fresh citrus fragrance with lemon and bergamot"
        embedding = encoder.encode(text)
        
        assert len(embedding) == 128
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Test reproducibility
        embedding2 = encoder.encode(text)
        assert embedding == embedding2
    
    def test_text_encoder_semantic_features(self):
        """Test text encoder semantic feature extraction."""
        encoder = TextEncoder(embedding_dim=200)
        
        # Test fragrance category keywords
        citrus_text = "bright citrus lemon"
        floral_text = "romantic floral rose"
        
        citrus_embedding = encoder.encode(citrus_text)
        floral_embedding = encoder.encode(floral_text)
        
        assert len(citrus_embedding) == 200
        assert len(floral_embedding) == 200
        
        # Embeddings should be different
        assert citrus_embedding != floral_embedding
    
    def test_molecule_encoder(self):
        """Test molecule encoder functionality.""" 
        encoder = MoleculeEncoder(embedding_dim=256)
        
        # Test with valid SMILES
        smiles = "CC(C)=CCCC(C)=CCO"  # Geraniol
        embedding = encoder.encode(smiles)
        
        assert len(embedding) == 256
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Test reproducibility
        embedding2 = encoder.encode(smiles)
        assert embedding == embedding2
    
    def test_molecule_encoder_structural_features(self):
        """Test molecule encoder structural pattern recognition."""
        encoder = MoleculeEncoder(embedding_dim=200)
        
        # Test different structural patterns
        carbonyl_mol = "CC(=O)CC"      # Contains C=O
        aromatic_mol = "c1ccccc1"      # Aromatic ring
        simple_mol = "CCC"             # Simple chain
        
        carbonyl_emb = encoder.encode(carbonyl_mol)
        aromatic_emb = encoder.encode(aromatic_mol)
        simple_emb = encoder.encode(simple_mol)
        
        # All should be different
        assert carbonyl_emb != aromatic_emb
        assert aromatic_emb != simple_emb
        assert simple_emb != carbonyl_emb
    
    def test_olfactory_encoder_dict_input(self):
        """Test olfactory encoder with dictionary input."""
        encoder = OlfactoryEncoder(embedding_dim=100)
        
        olfactory_profile = {
            "intensity": 0.8,
            "longevity": 0.6,
            "freshness": 0.9,
            "warmth": 0.3
        }
        
        embedding = encoder.encode(olfactory_profile)
        
        assert len(embedding) == 100
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_olfactory_encoder_string_input(self):
        """Test olfactory encoder with string input."""
        encoder = OlfactoryEncoder(embedding_dim=100)
        
        olfactory_description = "Intense and long-lasting with fresh top notes"
        embedding = encoder.encode(olfactory_description)
        
        assert len(embedding) == 100
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_projection_head(self):
        """Test projection head functionality."""
        embedding_dim = 64
        projection_head = ProjectionHead(embedding_dim, num_layers=2)
        
        # Test forward pass
        input_embedding = [random.gauss(0, 1) for _ in range(embedding_dim)]
        projected = projection_head.forward(input_embedding)
        
        assert len(projected) == embedding_dim
        
        # Output should be normalized
        magnitude = math.sqrt(sum(x*x for x in projected))
        assert abs(magnitude - 1.0) < 1e-10
        
        # Test reproducibility
        projected2 = projection_head.forward(input_embedding)
        assert projected == projected2


class TestMultimodalEncoder:
    """Test multimodal encoder functionality."""
    
    def test_encoder_initialization(self):
        """Test multimodal encoder initialization."""
        encoder = MultimodalEncoder(
            embedding_dim=256,
            temperature=0.05,
            projection_layers=3
        )
        
        assert encoder.embedding_dim == 256
        assert encoder.temperature == 0.05
        assert encoder.projection_layers == 3
        
        # Check sub-encoders are initialized
        assert isinstance(encoder.text_encoder, TextEncoder)
        assert isinstance(encoder.molecule_encoder, MoleculeEncoder)
        assert isinstance(encoder.olfactory_encoder, OlfactoryEncoder)
        assert isinstance(encoder.projection_head, ProjectionHead)
    
    def test_encode_text_modality(self):
        """Test text modality encoding."""
        encoder = MultimodalEncoder(embedding_dim=128)
        
        text = "Fresh oceanic fragrance"
        embedding = encoder.encode_modality(text, ModalityType.TEXT)
        
        assert embedding.modality_type == ModalityType.TEXT
        assert len(embedding.embedding) == 128
        assert 0.0 <= embedding.confidence <= 1.0
        
        # Embedding should be normalized
        magnitude = math.sqrt(sum(x*x for x in embedding.embedding))
        assert abs(magnitude - 1.0) < 1e-10
    
    def test_encode_molecule_modality(self):
        """Test molecule modality encoding."""
        encoder = MultimodalEncoder(embedding_dim=128)
        
        smiles = "CC(C)=CCCC(C)=CCO"
        embedding = encoder.encode_modality(smiles, ModalityType.MOLECULE)
        
        assert embedding.modality_type == ModalityType.MOLECULE
        assert len(embedding.embedding) == 128
        assert 0.0 <= embedding.confidence <= 1.0
    
    def test_encode_olfactory_modality(self):
        """Test olfactory modality encoding."""
        encoder = MultimodalEncoder(embedding_dim=128)
        
        olfactory_data = {"intensity": 0.8, "freshness": 0.9}
        embedding = encoder.encode_modality(olfactory_data, ModalityType.OLFACTORY)
        
        assert embedding.modality_type == ModalityType.OLFACTORY
        assert len(embedding.embedding) == 128
        assert 0.0 <= embedding.confidence <= 1.0
    
    def test_similarity_computation(self):
        """Test embedding similarity computation."""
        encoder = MultimodalEncoder()
        
        # Test identical embeddings
        emb1 = [1.0, 0.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0, 0.0]
        similarity = encoder._compute_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 1e-10
        
        # Test orthogonal embeddings
        emb3 = [0.0, 1.0, 0.0, 0.0]
        similarity_orth = encoder._compute_similarity(emb1, emb3)
        assert abs(similarity_orth - 0.0) < 1e-10
        
        # Test opposite embeddings
        emb4 = [-1.0, 0.0, 0.0, 0.0]
        similarity_opp = encoder._compute_similarity(emb1, emb4)
        assert abs(similarity_opp - (-1.0)) < 1e-10
    
    def test_contrastive_examples_creation(self):
        """Test contrastive examples creation from multimodal data."""
        encoder = MultimodalEncoder(embedding_dim=64)
        
        multimodal_data = [
            {
                "text": "Fresh citrus fragrance",
                "molecule": "CC(C)=CCCC(C)=CCO",
                "olfactory": {"intensity": 0.8, "freshness": 0.9}
            },
            {
                "text": "Warm woody scent",
                "molecule": "CC12CCC(CC1=CCC2=O)C(C)(C)C",
                "olfactory": {"intensity": 0.6, "warmth": 0.8}
            },
            {
                "text": "Floral rose bouquet",
                "molecule": "COC1=CC=C(C=C1)C=O",
                "olfactory": {"intensity": 0.7, "floral": 0.9}
            }
        ]
        
        examples = encoder.create_contrastive_examples(
            multimodal_data, 
            num_negatives=2
        )
        
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, ContrastiveExample)
            assert len(example.positive_pairs) > 0
            assert len(example.negative_pairs) > 0
            
            # Check that positive pairs are from same data point
            # Check that negative pairs are from different data points
            for pos_pair in example.positive_pairs:
                assert pos_pair[0].modality_type != pos_pair[1].modality_type
            
            for neg_pair in example.negative_pairs:
                # Negative pairs should have same modality as anchor
                assert neg_pair[0].modality_type == neg_pair[1].modality_type
    
    def test_contrastive_loss_computation(self):
        """Test contrastive loss computation."""
        encoder = MultimodalEncoder(temperature=0.1)
        
        # Create simple contrastive examples
        anchor = ModalityEmbedding(ModalityType.TEXT, [1.0, 0.0])
        positive = ModalityEmbedding(ModalityType.MOLECULE, [0.9, 0.436])  # Similar to anchor
        negative = ModalityEmbedding(ModalityType.MOLECULE, [0.0, 1.0])    # Orthogonal to anchor
        
        example = ContrastiveExample(
            anchor=anchor,
            positive_pairs=[(anchor, positive)],
            negative_pairs=[(anchor, negative)]
        )
        
        loss_info = encoder.compute_contrastive_loss([example])
        
        assert 'total_loss' in loss_info
        assert 'positive_similarity_avg' in loss_info
        assert 'negative_similarity_avg' in loss_info
        assert 'similarity_gap' in loss_info
        
        assert loss_info['total_loss'] >= 0.0
        assert loss_info['positive_similarity_avg'] > loss_info['negative_similarity_avg']
        assert loss_info['similarity_gap'] > 0.0
    
    def test_contrastive_training(self):
        """Test contrastive training process."""
        encoder = MultimodalEncoder(embedding_dim=32)
        
        multimodal_data = [
            {
                "text": "Fresh citrus",
                "molecule": "CC=O",
                "olfactory": {"intensity": 0.8}
            },
            {
                "text": "Warm spice",
                "molecule": "CCC=O",
                "olfactory": {"intensity": 0.6}
            }
        ]
        
        training_report = encoder.train_contrastive(
            multimodal_data,
            num_epochs=5,
            batch_size=2,
            learning_rate=0.01
        )
        
        assert training_report['training_completed']
        assert training_report['num_epochs'] == 5
        assert training_report['final_loss'] >= 0.0
        assert 0.0 <= training_report['final_alignment_score'] <= 1.0
        assert len(training_report['loss_history']) == 5
        assert training_report['num_examples'] > 0
    
    def test_aligned_representation_generation(self):
        """Test aligned multimodal representation generation."""
        encoder = MultimodalEncoder(embedding_dim=64)
        
        # Train briefly to set up the encoder
        multimodal_data = [
            {
                "text": "Fresh citrus fragrance",
                "molecule": "CC(C)=CCCC(C)=CCO"
            }
        ]
        encoder.train_contrastive(multimodal_data, num_epochs=2)
        
        # Generate aligned representation
        text_prompt = "Bright lemon fragrance"
        reference_molecule = "CC(C)=CCCC(C)=CCO"
        olfactory_profile = {"intensity": 0.8, "freshness": 0.9}
        
        aligned_repr = encoder.generate_aligned_representation(
            text_prompt=text_prompt,
            reference_molecule=reference_molecule,
            olfactory_profile=olfactory_profile
        )
        
        assert 'unified_embedding' in aligned_repr
        assert 'modality_embeddings' in aligned_repr
        assert 'guidance_signals' in aligned_repr
        assert 'alignment_confidence' in aligned_repr
        assert 'generation_hints' in aligned_repr
        
        # Check unified embedding
        unified_emb = aligned_repr['unified_embedding']
        assert len(unified_emb) == 64
        
        # Check guidance signals structure
        guidance = aligned_repr['guidance_signals']
        assert 'structural_hints' in guidance
        assert 'property_targets' in guidance
        assert 'generation_temperature' in guidance
        assert 'diversity_encouragement' in guidance
        
        # Check alignment confidence
        assert 0.0 <= aligned_repr['alignment_confidence'] <= 1.0
        
        # Check generation hints
        hints = aligned_repr['generation_hints']
        assert isinstance(hints, list)
    
    def test_guidance_signal_extraction(self):
        """Test guidance signal extraction from embeddings."""
        encoder = MultimodalEncoder(embedding_dim=128)
        
        # Create unified embedding
        unified_emb = ModalityEmbedding(
            ModalityType.TEXT,
            [random.gauss(0, 0.1) for _ in range(128)]
        )
        
        text_prompt = "Intense woody fragrance with cedar and sandalwood notes"
        
        guidance = encoder._generate_guidance_signals(unified_emb, text_prompt)
        
        # Check structural hints
        structural = guidance['structural_hints']
        assert 'aromatic_preference' in structural
        assert 'functional_group_bias' in structural
        assert 'molecular_size_preference' in structural
        assert 'complexity_level' in structural
        
        assert 0.0 <= structural['aromatic_preference'] <= 1.0
        assert structural['molecular_size_preference'] in ['small', 'medium', 'large']
        assert 0.0 <= structural['complexity_level'] <= 1.0
        
        # Check property targets
        properties = guidance['property_targets']
        assert 'intensity' in properties
        assert 'longevity' in properties
        assert 'freshness' in properties
        assert 'warmth' in properties
        
        for prop_value in properties.values():
            assert 0.0 <= prop_value <= 1.0
        
        # Check generation parameters
        assert 0.1 <= guidance['generation_temperature'] <= 1.0
        assert 0.1 <= guidance['diversity_encouragement'] <= 0.9
    
    def test_training_statistics(self):
        """Test training statistics tracking."""
        encoder = MultimodalEncoder()
        
        # Run some training to generate statistics
        multimodal_data = [
            {"text": "fresh", "molecule": "CC"},
            {"text": "woody", "molecule": "CCC"}
        ]
        
        encoder.train_contrastive(multimodal_data, num_epochs=3)
        
        stats = encoder.get_training_statistics()
        
        assert 'contrastive_loss' in stats
        assert 'alignment_scores' in stats
        assert 'modality_usage' in stats
        assert 'similarity_statistics' in stats
        
        # Check loss statistics
        loss_stats = stats['contrastive_loss']
        assert 'current' in loss_stats
        assert 'average' in loss_stats
        assert 'trend' in loss_stats
        assert loss_stats['trend'] in ['improving', 'declining', 'stable', 'insufficient_data']
        
        # Check alignment statistics
        alignment_stats = stats['alignment_scores']
        assert 'current' in alignment_stats
        assert 'average' in alignment_stats
        assert 'best' in alignment_stats
        
        # Check similarity statistics
        sim_stats = stats['similarity_statistics']
        assert 'separation_quality' in sim_stats
        assert 0.0 <= sim_stats['separation_quality'] <= 1.0


class TestContrastiveLearningValidation:
    """Test contrastive learning validation."""
    
    @pytest.mark.asyncio
    async def test_validate_contrastive_learning_basic(self):
        """Test basic contrastive learning validation."""
        encoder = MultimodalEncoder(embedding_dim=32)
        
        test_data = [
            {
                "text": "Fresh citrus fragrance",
                "molecule": "CC=O"
            },
            {
                "text": "Woody cedar scent", 
                "molecule": "CCC"
            },
            {
                "text": "Floral rose bouquet",
                "molecule": "COC"
            }
        ]
        
        # Run validation with limited trials
        validation_results = await validate_contrastive_learning(
            encoder, test_data, num_validation_runs=2
        )
        
        assert 'alignment_scores' in validation_results
        assert 'cross_modal_retrieval' in validation_results
        assert 'generation_quality' in validation_results
        assert 'statistical_significance' in validation_results
        
        assert len(validation_results['alignment_scores']) == 2
        assert len(validation_results['cross_modal_retrieval']) == 2
        assert len(validation_results['generation_quality']) == 2
        
        # Check statistical significance structure
        sig_tests = validation_results['statistical_significance']
        for metric_name, stats in sig_tests.items():
            if isinstance(stats, dict):
                assert 'mean' in stats
                assert 'std' in stats
                assert 'significant' in stats
                assert isinstance(stats['significant'], bool)
    
    def test_cross_modal_alignment_quality(self):
        """Test cross-modal alignment quality measurement."""
        encoder = MultimodalEncoder(embedding_dim=64)
        
        # Create perfectly aligned example
        perfect_text = ModalityEmbedding(ModalityType.TEXT, [1.0, 0.0])
        perfect_mol = ModalityEmbedding(ModalityType.MOLECULE, [1.0, 0.0])  # Identical
        
        perfect_example = ContrastiveExample(
            anchor=perfect_text,
            positive_pairs=[(perfect_text, perfect_mol)],
            negative_pairs=[(perfect_text, ModalityEmbedding(ModalityType.MOLECULE, [0.0, 1.0]))]
        )
        
        perfect_alignment = encoder._evaluate_alignment([perfect_example])
        
        # Create poorly aligned example
        poor_text = ModalityEmbedding(ModalityType.TEXT, [1.0, 0.0])
        poor_mol = ModalityEmbedding(ModalityType.MOLECULE, [0.0, 1.0])  # Orthogonal
        
        poor_example = ContrastiveExample(
            anchor=poor_text,
            positive_pairs=[(poor_text, poor_mol)],
            negative_pairs=[(poor_text, ModalityEmbedding(ModalityType.MOLECULE, [0.9, 0.436]))]  # Actually more similar
        )
        
        poor_alignment = encoder._evaluate_alignment([poor_example])
        
        # Perfect alignment should be better than poor alignment
        assert perfect_alignment >= poor_alignment
    
    def test_generation_hint_extraction(self):
        """Test generation hint extraction accuracy."""
        encoder = MultimodalEncoder(embedding_dim=100)
        
        # Create embedding with known patterns
        test_embedding = [0.0] * 100
        test_embedding[50:60] = [0.8] * 10  # High aromatic preference
        
        unified_emb = ModalityEmbedding(ModalityType.TEXT, test_embedding)
        
        # Test aromatic preference extraction
        aromatic_pref = encoder._extract_aromatic_preference(test_embedding)
        assert aromatic_pref > 0.5  # Should detect high aromatic preference
        
        # Test functional group bias extraction
        func_groups = encoder._extract_functional_group_bias(test_embedding, "sweet vanilla dessert")
        assert func_groups['aldehyde'] > 0.2  # Should detect sweet keywords
        assert func_groups['ester'] > 0.1
        
        # Test size preference
        size_pref = encoder._extract_size_preference(test_embedding)
        assert size_pref in ['small', 'medium', 'large']
        
        # Test property prediction
        intensity = encoder._predict_intensity_target(test_embedding, "strong powerful bold fragrance")
        assert intensity > 0.7  # Should predict high intensity from keywords
        
        freshness = encoder._predict_freshness_target(test_embedding, "fresh crisp clean airy scent")
        assert freshness > 0.7  # Should predict high freshness from keywords


class TestEncoderRobustness:
    """Test encoder robustness and edge cases."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        encoder = MultimodalEncoder()
        
        # Test empty text
        text_emb = encoder.encode_modality("", ModalityType.TEXT)
        assert len(text_emb.embedding) == encoder.embedding_dim
        assert text_emb.confidence < 0.5  # Should have low confidence
        
        # Test empty molecule
        mol_emb = encoder.encode_modality("", ModalityType.MOLECULE)
        assert len(mol_emb.embedding) == encoder.embedding_dim
        assert mol_emb.confidence < 0.5
        
        # Test empty olfactory dict
        olf_emb = encoder.encode_modality({}, ModalityType.OLFACTORY)
        assert len(olf_emb.embedding) == encoder.embedding_dim
    
    def test_invalid_modality_handling(self):
        """Test handling of invalid modality types."""
        encoder = MultimodalEncoder()
        
        # This should raise ValueError for unsupported modality
        with pytest.raises(ValueError):
            # Create a fake modality type (not actually possible with enum, 
            # but test the error handling path)
            encoder.encode_modality("test", "invalid_modality")
    
    def test_large_input_handling(self):
        """Test handling of very large inputs."""
        encoder = MultimodalEncoder()
        
        # Test very long text
        long_text = "fragrant " * 1000
        text_emb = encoder.encode_modality(long_text, ModalityType.TEXT)
        assert len(text_emb.embedding) == encoder.embedding_dim
        
        # Test very long molecule
        long_smiles = "C" * 500
        mol_emb = encoder.encode_modality(long_smiles, ModalityType.MOLECULE)
        assert len(mol_emb.embedding) == encoder.embedding_dim
    
    def test_contrastive_loss_edge_cases(self):
        """Test contrastive loss computation edge cases."""
        encoder = MultimodalEncoder()
        
        # Test empty examples list
        loss_info = encoder.compute_contrastive_loss([])
        assert loss_info['total_loss'] == 0.0
        
        # Test example with no positive or negative pairs
        anchor = ModalityEmbedding(ModalityType.TEXT, [1.0, 0.0])
        empty_example = ContrastiveExample(
            anchor=anchor,
            positive_pairs=[],
            negative_pairs=[]
        )
        
        loss_info = encoder.compute_contrastive_loss([empty_example])
        assert loss_info['total_loss'] >= 0.0  # Should handle gracefully
    
    def test_similarity_edge_cases(self):
        """Test similarity computation edge cases."""
        encoder = MultimodalEncoder()
        
        # Test empty embeddings
        sim1 = encoder._compute_similarity([], [])
        assert sim1 == 0.0
        
        # Test mismatched lengths
        sim2 = encoder._compute_similarity([1.0, 0.0], [1.0])
        assert sim2 == 0.0
        
        # Test zero embeddings
        sim3 = encoder._compute_similarity([0.0, 0.0], [0.0, 0.0])
        assert sim3 == 0.0


@pytest.fixture
def sample_multimodal_encoder():
    """Fixture providing a sample multimodal encoder."""
    return MultimodalEncoder(embedding_dim=64, temperature=0.1, projection_layers=2)


@pytest.fixture
def sample_multimodal_data():
    """Fixture providing sample multimodal data."""
    return [
        {
            "text": "Fresh citrus fragrance with bright lemon notes",
            "molecule": "CC(C)=CCCC(C)=CCO",
            "olfactory": {"intensity": 0.8, "freshness": 0.9, "citrus": 0.7}
        },
        {
            "text": "Warm woody scent with cedar and sandalwood",
            "molecule": "CC12CCC(CC1=CCC2=O)C(C)(C)C",
            "olfactory": {"intensity": 0.6, "warmth": 0.8, "woody": 0.9}
        },
        {
            "text": "Elegant floral bouquet with rose and jasmine",
            "molecule": "COC1=CC=C(C=C1)C=O",
            "olfactory": {"intensity": 0.7, "floral": 0.9, "elegance": 0.8}
        },
        {
            "text": "Fresh oceanic breeze with marine accord",
            "molecule": "CC(C)=CCC=C(C)C",
            "olfactory": {"intensity": 0.5, "freshness": 1.0, "marine": 0.8}
        }
    ]