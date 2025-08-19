"""
Cross-Modal Contrastive Learning for Enhanced Molecular-Text Alignment

Implementation of CLIP-style contrastive learning for molecules and olfactory descriptors
to improve cross-modal understanding and generation accuracy.

Research Hypothesis: Contrastive learning between molecular structures, olfactory 
descriptors, and neural activation patterns will improve cross-modal understanding by 40%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import logging
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ContrastiveBatch:
    """Batch data for contrastive learning"""
    molecular_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    olfactory_descriptors: torch.Tensor
    similarity_matrix: torch.Tensor
    batch_size: int

class MolecularEncoder(nn.Module):
    """Molecular structure encoder with graph neural networks"""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph neural network layers
        self.atom_embedding = nn.Embedding(120, 64)  # 120 atom types
        self.bond_embedding = nn.Embedding(10, 32)   # 10 bond types
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(64 + 32, hidden_dim) for _ in range(3)
        ])
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, molecular_graphs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode molecular graphs to embeddings"""
        atom_features = molecular_graphs['atom_features']
        bond_features = molecular_graphs['bond_features']
        adjacency_matrix = molecular_graphs['adjacency']
        
        # Embed atoms and bonds
        atom_embeds = self.atom_embedding(atom_features)
        bond_embeds = self.bond_embedding(bond_features)
        
        # Message passing
        node_features = atom_embeds
        for layer in self.message_layers:
            # Aggregate neighbor features
            messages = torch.matmul(adjacency_matrix, node_features)
            # Combine with bond features (simplified)
            combined = torch.cat([messages, bond_embeds.mean(dim=1, keepdim=True).expand_as(messages)], dim=-1)
            node_features = F.relu(layer(combined))
        
        # Global pooling
        molecular_embedding = node_features.mean(dim=1)
        
        # Final projection
        return self.projection(molecular_embedding)

class OlfactoryEncoder(nn.Module):
    """Olfactory descriptor encoder with attention mechanisms"""
    
    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, descriptor_tokens: torch.Tensor) -> torch.Tensor:
        """Encode olfactory descriptors"""
        embeddings = self.embedding(descriptor_tokens)
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        
        # Global average pooling
        pooled = attended.mean(dim=1)
        return self.projection(pooled)

class TextEncoder(nn.Module):
    """Text encoder using pre-trained transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", output_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)
        
    def forward(self, text_inputs: List[str]) -> torch.Tensor:
        """Encode text descriptions"""
        tokenized = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.encoder(**tokenized)
            text_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return self.projection(text_embeddings)

class ContrastiveLearningModel(nn.Module):
    """Main contrastive learning model for cross-modal alignment"""
    
    def __init__(self, embedding_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Encoders
        self.molecular_encoder = MolecularEncoder(output_dim=embedding_dim)
        self.text_encoder = TextEncoder(output_dim=embedding_dim)
        self.olfactory_encoder = OlfactoryEncoder(output_dim=embedding_dim)
        
        # Projection heads for contrastive learning
        self.mol_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def forward(self, batch: ContrastiveBatch) -> Dict[str, torch.Tensor]:
        """Forward pass for contrastive learning"""
        # Encode all modalities
        mol_embeddings = self.molecular_encoder(batch.molecular_embeddings)
        text_embeddings = self.text_encoder(batch.text_embeddings)
        olfactory_embeddings = self.olfactory_encoder(batch.olfactory_descriptors)
        
        # Apply projection heads
        mol_proj = F.normalize(self.mol_projection(mol_embeddings), dim=-1)
        text_proj = F.normalize(self.text_projection(text_embeddings), dim=-1)
        olfactory_proj = F.normalize(olfactory_embeddings, dim=-1)
        
        # Compute similarities
        temperature = torch.exp(self.log_temperature)
        
        # Molecule-Text contrastive loss
        mol_text_sim = torch.matmul(mol_proj, text_proj.T) / temperature
        mol_text_labels = torch.arange(mol_text_sim.size(0), device=mol_text_sim.device)
        
        mol_text_loss = (
            F.cross_entropy(mol_text_sim, mol_text_labels) +
            F.cross_entropy(mol_text_sim.T, mol_text_labels)
        ) / 2
        
        # Molecule-Olfactory contrastive loss
        mol_olfactory_sim = torch.matmul(mol_proj, olfactory_proj.T) / temperature
        mol_olfactory_loss = (
            F.cross_entropy(mol_olfactory_sim, mol_text_labels) +
            F.cross_entropy(mol_olfactory_sim.T, mol_text_labels)
        ) / 2
        
        # Text-Olfactory contrastive loss
        text_olfactory_sim = torch.matmul(text_proj, olfactory_proj.T) / temperature
        text_olfactory_loss = (
            F.cross_entropy(text_olfactory_sim, mol_text_labels) +
            F.cross_entropy(text_olfactory_sim.T, mol_text_labels)
        ) / 2
        
        # Total contrastive loss
        total_loss = mol_text_loss + mol_olfactory_loss + text_olfactory_loss
        
        return {
            'total_loss': total_loss,
            'mol_text_loss': mol_text_loss,
            'mol_olfactory_loss': mol_olfactory_loss,
            'text_olfactory_loss': text_olfactory_loss,
            'mol_embeddings': mol_proj,
            'text_embeddings': text_proj,
            'olfactory_embeddings': olfactory_proj,
            'temperature': temperature
        }

class ContrastiveTrainer:
    """Trainer for contrastive learning model"""
    
    def __init__(self, model: ContrastiveLearningModel, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
    def train_step(self, batch: ContrastiveBatch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(batch)
        loss = outputs['total_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': loss.item(),
            'mol_text_loss': outputs['mol_text_loss'].item(),
            'mol_olfactory_loss': outputs['mol_olfactory_loss'].item(),
            'text_olfactory_loss': outputs['text_olfactory_loss'].item(),
            'temperature': outputs['temperature'].item()
        }
    
    def evaluate(self, eval_batches: List[ContrastiveBatch]) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        retrieval_accuracies = []
        
        with torch.no_grad():
            for batch in eval_batches:
                outputs = self.model(batch)
                total_loss += outputs['total_loss'].item()
                
                # Calculate retrieval accuracy
                mol_embeddings = outputs['mol_embeddings']
                text_embeddings = outputs['text_embeddings']
                
                # Compute similarity matrix
                similarities = torch.matmul(mol_embeddings, text_embeddings.T)
                
                # Check if correct pairs are ranked highest
                correct_predictions = (similarities.argmax(dim=1) == torch.arange(similarities.size(0))).float()
                retrieval_accuracies.append(correct_predictions.mean().item())
        
        return {
            'eval_loss': total_loss / len(eval_batches),
            'retrieval_accuracy': np.mean(retrieval_accuracies)
        }

class ContrastiveDataPreprocessor:
    """Data preprocessing for contrastive learning"""
    
    def __init__(self):
        self.molecular_featurizer = self._init_molecular_featurizer()
        self.text_processor = self._init_text_processor()
        self.olfactory_vocab = self._build_olfactory_vocab()
        
    def _init_molecular_featurizer(self):
        """Initialize molecular featurization"""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors, rdMolChemicalFeatures
            return {
                'rdkit': Chem,
                'descriptors': rdMolDescriptors,
                'features': rdMolChemicalFeatures
            }
        except ImportError:
            logger.warning("RDKit not available, using mock featurizer")
            return None
    
    def _init_text_processor(self):
        """Initialize text processing"""
        return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def _build_olfactory_vocab(self) -> Dict[str, int]:
        """Build vocabulary for olfactory descriptors"""
        common_descriptors = [
            'floral', 'fruity', 'woody', 'citrus', 'spicy', 'sweet', 'fresh',
            'musky', 'herbal', 'vanilla', 'rose', 'jasmine', 'sandalwood',
            'bergamot', 'lavender', 'mint', 'pine', 'ocean', 'green',
            'powdery', 'creamy', 'smoky', 'earthy', 'metallic', 'aquatic'
        ]
        return {desc: i for i, desc in enumerate(common_descriptors)}
    
    def process_molecular_data(self, smiles: List[str]) -> Dict[str, torch.Tensor]:
        """Process molecular SMILES to graph representations"""
        if self.molecular_featurizer is None:
            # Mock processing for systems without RDKit
            batch_size = len(smiles)
            return {
                'atom_features': torch.randint(0, 120, (batch_size, 50)),
                'bond_features': torch.randint(0, 10, (batch_size, 50)),
                'adjacency': torch.rand(batch_size, 50, 50) > 0.8
            }
        
        # Real RDKit processing would go here
        processed_graphs = []
        for smile in smiles:
            try:
                mol = self.molecular_featurizer['rdkit'].MolFromSmiles(smile)
                if mol is not None:
                    # Extract atom and bond features
                    atom_features = []
                    bond_features = []
                    adjacency = []
                    
                    # Process atoms
                    for atom in mol.GetAtoms():
                        atom_features.append(atom.GetAtomicNum())
                    
                    # Process bonds
                    for bond in mol.GetBonds():
                        bond_features.append(int(bond.GetBondType()))
                    
                    # Create adjacency matrix
                    adj_matrix = torch.zeros(len(atom_features), len(atom_features))
                    for bond in mol.GetBonds():
                        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        adj_matrix[i, j] = adj_matrix[j, i] = 1
                    
                    processed_graphs.append({
                        'atom_features': torch.tensor(atom_features),
                        'bond_features': torch.tensor(bond_features),
                        'adjacency': adj_matrix
                    })
                else:
                    # Invalid SMILES, use padding
                    processed_graphs.append({
                        'atom_features': torch.zeros(1),
                        'bond_features': torch.zeros(1),
                        'adjacency': torch.zeros(1, 1)
                    })
            except Exception as e:
                logger.warning(f"Error processing SMILES {smile}: {e}")
                processed_graphs.append({
                    'atom_features': torch.zeros(1),
                    'bond_features': torch.zeros(1),
                    'adjacency': torch.zeros(1, 1)
                })
        
        # Batch and pad graphs
        max_atoms = max(graph['atom_features'].size(0) for graph in processed_graphs)
        
        batched_atom_features = []
        batched_bond_features = []
        batched_adjacency = []
        
        for graph in processed_graphs:
            n_atoms = graph['atom_features'].size(0)
            
            # Pad atom features
            padded_atoms = torch.zeros(max_atoms)
            padded_atoms[:n_atoms] = graph['atom_features']
            batched_atom_features.append(padded_atoms)
            
            # Pad bond features (simplified)
            padded_bonds = torch.zeros(max_atoms)
            n_bonds = min(len(graph['bond_features']), max_atoms)
            if n_bonds > 0:
                padded_bonds[:n_bonds] = graph['bond_features'][:n_bonds]
            batched_bond_features.append(padded_bonds)
            
            # Pad adjacency matrix
            padded_adj = torch.zeros(max_atoms, max_atoms)
            padded_adj[:n_atoms, :n_atoms] = graph['adjacency']
            batched_adjacency.append(padded_adj)
        
        return {
            'atom_features': torch.stack(batched_atom_features).long(),
            'bond_features': torch.stack(batched_bond_features).long(),
            'adjacency': torch.stack(batched_adjacency).float()
        }
    
    def process_olfactory_descriptors(self, descriptors: List[List[str]]) -> torch.Tensor:
        """Process olfactory descriptor lists to token tensors"""
        max_descriptors = max(len(desc_list) for desc_list in descriptors)
        
        tokenized_descriptors = []
        for desc_list in descriptors:
            tokens = []
            for desc in desc_list:
                if desc.lower() in self.olfactory_vocab:
                    tokens.append(self.olfactory_vocab[desc.lower()])
                else:
                    tokens.append(0)  # Unknown token
            
            # Pad to max length
            while len(tokens) < max_descriptors:
                tokens.append(0)  # Padding token
            
            tokenized_descriptors.append(tokens[:max_descriptors])
        
        return torch.tensor(tokenized_descriptors, dtype=torch.long)

def create_contrastive_learning_system():
    """Factory function to create complete contrastive learning system"""
    model = ContrastiveLearningModel(embedding_dim=256, temperature=0.07)
    trainer = ContrastiveTrainer(model, learning_rate=1e-4)
    preprocessor = ContrastiveDataPreprocessor()
    
    return {
        'model': model,
        'trainer': trainer,
        'preprocessor': preprocessor
    }

# Experimental validation functions
def run_contrastive_experiment(training_data: List[Dict], validation_data: List[Dict]) -> Dict:
    """Run complete contrastive learning experiment"""
    logger.info("Starting contrastive learning experiment")
    
    # Initialize system
    system = create_contrastive_learning_system()
    model = system['model']
    trainer = system['trainer']
    preprocessor = system['preprocessor']
    
    # Process training data
    train_batches = []
    for data_point in training_data:
        molecular_graphs = preprocessor.process_molecular_data([data_point['smiles']])
        olfactory_tokens = preprocessor.process_olfactory_descriptors([data_point['descriptors']])
        
        batch = ContrastiveBatch(
            molecular_embeddings=molecular_graphs,
            text_embeddings=[data_point['text_description']],
            olfactory_descriptors=olfactory_tokens,
            similarity_matrix=torch.eye(1),
            batch_size=1
        )
        train_batches.append(batch)
    
    # Training loop
    training_metrics = []
    for epoch in range(100):  # Reduced for demo
        epoch_losses = []
        for batch in train_batches:
            metrics = trainer.train_step(batch)
            epoch_losses.append(metrics['total_loss'])
        
        avg_loss = np.mean(epoch_losses)
        training_metrics.append({'epoch': epoch, 'loss': avg_loss})
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    # Evaluation
    val_batches = []
    for data_point in validation_data:
        molecular_graphs = preprocessor.process_molecular_data([data_point['smiles']])
        olfactory_tokens = preprocessor.process_olfactory_descriptors([data_point['descriptors']])
        
        batch = ContrastiveBatch(
            molecular_embeddings=molecular_graphs,
            text_embeddings=[data_point['text_description']],
            olfactory_descriptors=olfactory_tokens,
            similarity_matrix=torch.eye(1),
            batch_size=1
        )
        val_batches.append(batch)
    
    eval_metrics = trainer.evaluate(val_batches)
    
    logger.info(f"Final Evaluation - Loss: {eval_metrics['eval_loss']:.4f}, "
                f"Retrieval Accuracy: {eval_metrics['retrieval_accuracy']:.4f}")
    
    return {
        'training_metrics': training_metrics,
        'evaluation_metrics': eval_metrics,
        'model': model,
        'final_loss': eval_metrics['eval_loss'],
        'retrieval_accuracy': eval_metrics['retrieval_accuracy']
    }

if __name__ == "__main__":
    # Demo with synthetic data
    synthetic_training_data = [
        {
            'smiles': 'CCO',
            'text_description': 'Fresh alcoholic scent with clean notes',
            'descriptors': ['fresh', 'clean', 'alcoholic']
        },
        {
            'smiles': 'CC(C)CCO',
            'text_description': 'Sweet floral fragrance with rosy undertones',
            'descriptors': ['sweet', 'floral', 'rosy']
        }
    ]
    
    synthetic_validation_data = [
        {
            'smiles': 'CCCO',
            'text_description': 'Light alcoholic scent',
            'descriptors': ['light', 'alcoholic']
        }
    ]
    
    results = run_contrastive_experiment(synthetic_training_data, synthetic_validation_data)
    print(f"Experiment completed with final accuracy: {results['retrieval_accuracy']:.3f}")