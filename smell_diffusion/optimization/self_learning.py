"""
Self-Learning Molecular Optimization System

Advanced AI-driven optimization that learns from generation patterns:
- Reinforcement learning for molecular design
- Evolutionary algorithms for structure optimization
- Adaptive prompt understanding
- Self-improving generation quality
"""

import time
import random
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import hashlib

try:
    import numpy as np
except ImportError:
    # Advanced numerical fallback
    class AdvancedNumPy:
        @staticmethod
        def array(x): return x
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): 
            mean_val = sum(x) / len(x) if x else 0
            variance = sum((i - mean_val) ** 2 for i in x) / len(x) if x else 0
            return variance ** 0.5
        @staticmethod
        def exp(x): return math.exp(x) if isinstance(x, (int, float)) else [math.exp(i) for i in x]
        @staticmethod
        def log(x): return math.log(x) if isinstance(x, (int, float)) else [math.log(i) for i in x]
        @staticmethod
        def random(): return random
        @staticmethod
        def choice(items, p=None): return random.choice(items)
        @staticmethod
        def argmax(x): return x.index(max(x)) if x else 0
        @staticmethod
        def softmax(x):
            exp_x = [math.exp(i - max(x)) for i in x]
            sum_exp_x = sum(exp_x)
            return [i / sum_exp_x for i in exp_x]
    np = AdvancedNumPy()

from ..core.molecule import Molecule
from ..core.smell_diffusion import SmellDiffusion
from ..utils.logging import SmellDiffusionLogger, performance_monitor


@dataclass
class LearningMetrics:
    """Metrics for self-learning optimization."""
    generation_quality_trend: List[float]
    prompt_understanding_score: float
    optimization_effectiveness: float
    convergence_rate: float
    exploration_exploitation_ratio: float
    learning_stability: float


@dataclass
class OptimizationAction:
    """Action taken by the optimization system."""
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    timestamp: float


class ReinforcementLearningOptimizer:
    """RL-based molecular optimization system."""
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action value table
        self.experience_buffer = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.logger = SmellDiffusionLogger("rl_optimizer")
        
        # Define action space
        self.actions = [
            "increase_diversity",
            "improve_safety",
            "enhance_relevance", 
            "optimize_properties",
            "explore_novel_structures",
            "refine_existing_structures"
        ]
        
    def get_state(self, prompt: str, generation_history: List[Dict[str, Any]]) -> str:
        """Convert current context to state representation."""
        
        # Extract key features from prompt
        prompt_features = self._extract_prompt_features(prompt)
        
        # Extract features from recent generation history
        history_features = self._extract_history_features(generation_history)
        
        # Create state representation
        state = f"{prompt_features}_{history_features}"
        return hashlib.md5(state.encode()).hexdigest()[:16]
    
    def _extract_prompt_features(self, prompt: str) -> str:
        """Extract relevant features from prompt."""
        
        prompt_lower = prompt.lower()
        
        # Scent categories
        categories = {
            'citrus': any(word in prompt_lower for word in ['citrus', 'lemon', 'orange', 'bergamot']),
            'floral': any(word in prompt_lower for word in ['floral', 'rose', 'jasmine', 'lavender']),
            'woody': any(word in prompt_lower for word in ['woody', 'cedar', 'sandalwood', 'oak']),
            'fresh': any(word in prompt_lower for word in ['fresh', 'clean', 'aquatic', 'marine']),
        }
        
        # Complexity indicators
        complexity = {
            'simple': len(prompt.split()) < 5,
            'moderate': 5 <= len(prompt.split()) < 10,
            'complex': len(prompt.split()) >= 10
        }
        
        # Emotional indicators
        emotions = {
            'romantic': any(word in prompt_lower for word in ['romantic', 'sensual', 'intimate']),
            'energetic': any(word in prompt_lower for word in ['energetic', 'vibrant', 'dynamic']),
            'calming': any(word in prompt_lower for word in ['calming', 'relaxing', 'peaceful'])
        }
        
        # Combine features
        features = []
        for category, present in categories.items():
            if present:
                features.append(category)
        
        for level, present in complexity.items():
            if present:
                features.append(f"complexity_{level}")
                
        for emotion, present in emotions.items():
            if present:
                features.append(f"emotion_{emotion}")
        
        return "_".join(sorted(features)) or "generic"
    
    def _extract_history_features(self, history: List[Dict[str, Any]]) -> str:
        """Extract features from generation history."""
        
        if not history:
            return "no_history"
        
        # Analyze recent performance
        recent_history = history[-5:]  # Last 5 generations
        
        quality_scores = [h.get('quality_score', 0.5) for h in recent_history]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        safety_scores = [h.get('safety_score', 50) for h in recent_history]
        avg_safety = sum(safety_scores) / len(safety_scores)
        
        # Categorize performance
        if avg_quality > 0.8:
            quality_level = "high"
        elif avg_quality > 0.6:
            quality_level = "medium"
        else:
            quality_level = "low"
            
        if avg_safety > 80:
            safety_level = "high"
        elif avg_safety > 60:
            safety_level = "medium"
        else:
            safety_level = "low"
        
        return f"quality_{quality_level}_safety_{safety_level}"
    
    def select_action(self, state: str) -> str:
        """Select action using epsilon-greedy policy."""
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(self.actions)
            self.logger.logger.debug(f"Exploration: selected action {action}")
            return action
        
        # Exploit: best known action
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        best_action = max(self.q_table[state], key=self.q_table[state].get)
        self.logger.logger.debug(f"Exploitation: selected action {best_action}")
        return best_action
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning rule."""
        
        # Initialize Q-table entries if needed
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.logger.logger.debug(f"Updated Q({state}, {action}): {current_q:.3f} -> {new_q:.3f}")
    
    def calculate_reward(self, generated_molecules: List[Molecule], 
                        target_criteria: Dict[str, float]) -> float:
        """Calculate reward based on generation quality."""
        
        if not generated_molecules:
            return -1.0
        
        valid_molecules = [mol for mol in generated_molecules if mol and mol.is_valid]
        if not valid_molecules:
            return -0.5
        
        # Multi-objective reward calculation
        rewards = []
        
        # Validity reward
        validity_reward = len(valid_molecules) / len(generated_molecules)
        rewards.append(validity_reward * target_criteria.get('validity_weight', 0.3))
        
        # Safety reward
        safety_scores = []
        for mol in valid_molecules:
            try:
                safety_profile = mol.get_safety_profile()
                safety_scores.append(safety_profile.score / 100.0)
            except:
                safety_scores.append(0.0)
        
        safety_reward = np.mean(safety_scores) if safety_scores else 0.0
        rewards.append(safety_reward * target_criteria.get('safety_weight', 0.3))
        
        # Diversity reward
        diversity_reward = self._calculate_diversity_reward(valid_molecules)
        rewards.append(diversity_reward * target_criteria.get('diversity_weight', 0.2))
        
        # Novelty reward
        novelty_reward = self._calculate_novelty_reward(valid_molecules)
        rewards.append(novelty_reward * target_criteria.get('novelty_weight', 0.2))
        
        total_reward = sum(rewards)
        return max(-1.0, min(1.0, total_reward))  # Clamp to [-1, 1]
    
    def _calculate_diversity_reward(self, molecules: List[Molecule]) -> float:
        """Calculate reward based on molecular diversity."""
        
        if len(molecules) < 2:
            return 0.0
        
        unique_smiles = set(mol.smiles for mol in molecules if mol.smiles)
        return len(unique_smiles) / len(molecules)
    
    def _calculate_novelty_reward(self, molecules: List[Molecule]) -> float:
        """Calculate reward based on molecular novelty."""
        
        novelty_scores = []
        
        for mol in molecules:
            # Simple novelty metric based on structure complexity
            smiles = mol.smiles
            novelty_factors = [
                len(set(smiles)) / len(smiles) if smiles else 0,  # Character diversity
                smiles.count('=') / max(len(smiles), 1),  # Double bonds
                smiles.count('#') / max(len(smiles), 1),  # Triple bonds
                smiles.count('(') / max(len(smiles), 1),  # Branching
            ]
            novelty_scores.append(sum(novelty_factors) / len(novelty_factors))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0


class EvolutionaryOptimizer:
    """Evolutionary algorithm for molecular structure optimization."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation_count = 0
        self.population_history = []
        self.logger = SmellDiffusionLogger("evolutionary_optimizer")
        
    def optimize_population(self, seed_molecules: List[Molecule], 
                          fitness_function: Callable[[Molecule], float],
                          generations: int = 10) -> List[Molecule]:
        """Optimize molecular population using evolutionary algorithm."""
        
        self.logger.logger.info(f"Starting evolutionary optimization: {generations} generations")
        
        # Initialize population
        population = self._initialize_population(seed_molecules)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness_function(mol) for mol in population]
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(selected)
            
            # Mutation
            mutated = self._mutation(offspring)
            
            # Replace population
            population = self._replacement(population, mutated, fitness_scores)
            
            # Track progress
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            
            self.logger.logger.info(
                f"Generation {generation + 1}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}"
            )
            
            self.population_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'population_size': len(population)
            })
        
        # Return best molecules
        final_fitness = [fitness_function(mol) for mol in population]
        best_indices = sorted(range(len(population)), key=lambda i: final_fitness[i], reverse=True)
        
        return [population[i] for i in best_indices[:min(10, len(population))]]
    
    def _initialize_population(self, seed_molecules: List[Molecule]) -> List[Molecule]:
        """Initialize population from seed molecules."""
        
        population = seed_molecules[:self.population_size]
        
        # Fill remaining slots with variations
        while len(population) < self.population_size:
            # Create variations of existing molecules
            base_mol = random.choice(seed_molecules)
            varied_mol = self._create_variation(base_mol)
            population.append(varied_mol)
        
        return population
    
    def _create_variation(self, molecule: Molecule) -> Molecule:
        """Create a variation of the input molecule."""
        
        # Simple SMILES mutation for demonstration
        smiles = molecule.smiles
        
        # Random mutations
        if random.random() < 0.3 and 'C' in smiles:
            # Add a carbon
            smiles = smiles.replace('C', 'CC', 1)
        elif random.random() < 0.2 and 'CC' in smiles:
            # Add branching
            smiles = smiles.replace('CC', 'C(C)C', 1)
        elif random.random() < 0.1 and 'C=C' not in smiles:
            # Add double bond
            smiles = smiles.replace('CC', 'C=C', 1)
        
        return Molecule(smiles, description=f"Evolved from: {molecule.description}")
    
    def _selection(self, population: List[Molecule], 
                  fitness_scores: List[float]) -> List[Molecule]:
        """Tournament selection."""
        
        selected = []
        tournament_size = 3
        
        for _ in range(len(population) // 2):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)), tournament_size)
            winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[winner_index])
        
        return selected
    
    def _crossover(self, parents: List[Molecule]) -> List[Molecule]:
        """Create offspring through crossover."""
        
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Simple crossover: combine SMILES strings
            child1_smiles, child2_smiles = self._smiles_crossover(parent1.smiles, parent2.smiles)
            
            child1 = Molecule(child1_smiles, description=f"Crossover: {parent1.description[:20]}")
            child2 = Molecule(child2_smiles, description=f"Crossover: {parent2.description[:20]}")
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _smiles_crossover(self, smiles1: str, smiles2: str) -> Tuple[str, str]:
        """Perform crossover on SMILES strings."""
        
        # Simple crossover at random points
        if len(smiles1) > 2 and len(smiles2) > 2:
            point1 = random.randint(1, len(smiles1) - 1)
            point2 = random.randint(1, len(smiles2) - 1)
            
            child1 = smiles1[:point1] + smiles2[point2:]
            child2 = smiles2[:point2] + smiles1[point1:]
            
            return child1, child2
        
        return smiles1, smiles2
    
    def _mutation(self, offspring: List[Molecule]) -> List[Molecule]:
        """Apply mutations to offspring."""
        
        mutated = []
        
        for mol in offspring:
            if random.random() < self.mutation_rate:
                mutated_mol = self._mutate_molecule(mol)
                mutated.append(mutated_mol)
            else:
                mutated.append(mol)
        
        return mutated
    
    def _mutate_molecule(self, molecule: Molecule) -> Molecule:
        """Apply random mutation to molecule."""
        
        smiles = molecule.smiles
        
        # Random mutations
        mutations = [
            lambda s: s.replace('C', 'N', 1) if 'C' in s and random.random() < 0.1 else s,
            lambda s: s.replace('CC', 'CO', 1) if 'CC' in s and random.random() < 0.2 else s,
            lambda s: s + 'C' if len(s) < 20 and random.random() < 0.3 else s,
            lambda s: s[:-1] if len(s) > 3 and random.random() < 0.1 else s,
        ]
        
        # Apply random mutation
        mutation = random.choice(mutations)
        mutated_smiles = mutation(smiles)
        
        return Molecule(mutated_smiles, description=f"Mutated: {molecule.description}")
    
    def _replacement(self, population: List[Molecule], offspring: List[Molecule],
                    fitness_scores: List[float]) -> List[Molecule]:
        """Replace population with offspring (elitist strategy)."""
        
        # Combine population and offspring
        combined = population + offspring
        
        # Evaluate all
        combined_fitness = fitness_scores + [0.5 for _ in offspring]  # Placeholder fitness
        
        # Select best individuals
        best_indices = sorted(
            range(len(combined)), 
            key=lambda i: combined_fitness[i], 
            reverse=True
        )
        
        return [combined[i] for i in best_indices[:self.population_size]]


class SelfLearningOptimizer:
    """Main self-learning optimization system combining RL and evolutionary approaches."""
    
    def __init__(self, base_generator: SmellDiffusion):
        self.base_generator = base_generator
        self.rl_optimizer = ReinforcementLearningOptimizer()
        self.evo_optimizer = EvolutionaryOptimizer()
        self.learning_metrics = LearningMetrics([], 0.0, 0.0, 0.0, 0.0, 0.0)
        self.optimization_history = []
        self.logger = SmellDiffusionLogger("self_learning_optimizer")
        
    @performance_monitor("self_learning_optimization")
    def optimize_generation(self, prompt: str, num_molecules: int = 10,
                          optimization_iterations: int = 5) -> Dict[str, Any]:
        """Perform self-learning optimization of molecular generation."""
        
        self.logger.logger.info(f"Starting self-learning optimization: {optimization_iterations} iterations")
        
        # Initialize
        best_molecules = []
        optimization_actions = []
        quality_progression = []
        
        for iteration in range(optimization_iterations):
            self.logger.logger.info(f"Optimization iteration {iteration + 1}/{optimization_iterations}")
            
            # Get current state
            state = self.rl_optimizer.get_state(prompt, self.optimization_history)
            
            # Select optimization action
            action = self.rl_optimizer.select_action(state)
            optimization_actions.append(action)
            
            # Apply optimization action
            optimized_generation_params = self._apply_optimization_action(action, prompt)
            
            # Generate molecules with optimization
            molecules = self._generate_with_optimization(
                prompt, num_molecules, optimized_generation_params
            )
            
            # Evaluate generation quality
            quality_metrics = self._evaluate_generation_quality(molecules, prompt)
            quality_progression.append(quality_metrics)
            
            # Calculate reward
            reward = self.rl_optimizer.calculate_reward(
                molecules, 
                {'validity_weight': 0.3, 'safety_weight': 0.3, 'diversity_weight': 0.2, 'novelty_weight': 0.2}
            )
            
            # Update Q-learning
            next_state = self.rl_optimizer.get_state(prompt, self.optimization_history + [quality_metrics])
            self.rl_optimizer.update_q_value(state, action, reward, next_state)
            
            # Apply evolutionary optimization if beneficial
            if quality_metrics.get('diversity_score', 0) < 0.5:
                evolved_molecules = self.evo_optimizer.optimize_population(
                    molecules, 
                    self._create_fitness_function(prompt),
                    generations=3
                )
                molecules.extend(evolved_molecules)
            
            # Update best molecules
            if not best_molecules or quality_metrics.get('overall_score', 0) > max(q.get('overall_score', 0) for q in quality_progression[:-1]):
                best_molecules = molecules[:num_molecules]
            
            # Record optimization step
            optimization_step = {
                'iteration': iteration + 1,
                'action': action,
                'quality_metrics': quality_metrics,
                'reward': reward,
                'num_molecules': len(molecules)
            }
            self.optimization_history.append(optimization_step)
        
        # Update learning metrics
        self._update_learning_metrics(quality_progression, optimization_actions)
        
        # Generate final report
        optimization_report = {
            'best_molecules': best_molecules,
            'quality_progression': quality_progression,
            'optimization_actions': optimization_actions,
            'learning_metrics': self.learning_metrics,
            'final_quality': quality_progression[-1] if quality_progression else {},
            'improvement': self._calculate_improvement(quality_progression),
            'recommendations': self._generate_optimization_recommendations()
        }
        
        self.logger.logger.info(f"Optimization complete: {optimization_report['improvement']:.2%} improvement")
        return optimization_report
    
    def _apply_optimization_action(self, action: str, prompt: str) -> Dict[str, Any]:
        """Apply optimization action to generation parameters."""
        
        params = {}
        
        if action == "increase_diversity":
            params['diversity_penalty'] = 0.8
            params['num_molecules'] = int(params.get('num_molecules', 5) * 1.5)
            
        elif action == "improve_safety":
            params['safety_filter'] = True
            params['min_safety_score'] = 70
            
        elif action == "enhance_relevance":
            params['guidance_scale'] = 9.0
            
        elif action == "optimize_properties":
            params['property_optimization'] = True
            
        elif action == "explore_novel_structures":
            params['exploration_factor'] = 0.4
            params['novelty_boost'] = True
            
        elif action == "refine_existing_structures":
            params['refinement_mode'] = True
            params['structure_constraints'] = 'moderate'
        
        return params
    
    def _generate_with_optimization(self, prompt: str, num_molecules: int, 
                                  params: Dict[str, Any]) -> List[Molecule]:
        """Generate molecules with optimization parameters."""
        
        try:
            # Apply parameters to base generator
            molecules = self.base_generator.generate(
                prompt=prompt,
                num_molecules=num_molecules,
                **{k: v for k, v in params.items() if k in ['safety_filter', 'guidance_scale', 'min_safety_score']}
            )
            
            if not isinstance(molecules, list):
                molecules = [molecules] if molecules else []
            
            return molecules
            
        except Exception as e:
            self.logger.log_error("optimized_generation", e)
            # Fallback to basic generation
            return self.base_generator.generate(prompt=prompt, num_molecules=num_molecules)
    
    def _evaluate_generation_quality(self, molecules: List[Molecule], 
                                   prompt: str) -> Dict[str, float]:
        """Comprehensive quality evaluation."""
        
        if not molecules:
            return {'overall_score': 0.0, 'validity_score': 0.0, 'safety_score': 0.0, 'diversity_score': 0.0}
        
        # Validity
        valid_molecules = [mol for mol in molecules if mol and mol.is_valid]
        validity_score = len(valid_molecules) / len(molecules)
        
        # Safety
        safety_scores = []
        for mol in valid_molecules:
            try:
                safety_profile = mol.get_safety_profile()
                safety_scores.append(safety_profile.score / 100.0)
            except:
                safety_scores.append(0.5)
        
        safety_score = np.mean(safety_scores) if safety_scores else 0.0
        
        # Diversity
        unique_smiles = set(mol.smiles for mol in valid_molecules if mol.smiles)
        diversity_score = len(unique_smiles) / len(molecules) if molecules else 0.0
        
        # Prompt relevance
        relevance_scores = []
        for mol in valid_molecules:
            try:
                relevance = self._calculate_prompt_relevance(mol, prompt)
                relevance_scores.append(relevance)
            except:
                relevance_scores.append(0.0)
        
        relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Overall score
        overall_score = (validity_score + safety_score + diversity_score + relevance_score) / 4.0
        
        return {
            'overall_score': overall_score,
            'validity_score': validity_score,
            'safety_score': safety_score,
            'diversity_score': diversity_score,
            'relevance_score': relevance_score
        }
    
    def _calculate_prompt_relevance(self, molecule: Molecule, prompt: str) -> float:
        """Calculate how relevant molecule is to prompt."""
        
        try:
            prompt_lower = prompt.lower()
            fragrance_notes = molecule.fragrance_notes
            all_notes = fragrance_notes.top + fragrance_notes.middle + fragrance_notes.base
            
            matches = sum(1 for note in all_notes if note in prompt_lower)
            return matches / max(len(all_notes), 1)
        except:
            return 0.0
    
    def _create_fitness_function(self, prompt: str) -> Callable[[Molecule], float]:
        """Create fitness function for evolutionary optimization."""
        
        def fitness_function(molecule: Molecule) -> float:
            if not molecule or not molecule.is_valid:
                return 0.0
            
            # Multi-objective fitness
            validity_score = 1.0 if molecule.is_valid else 0.0
            
            try:
                safety_score = molecule.get_safety_profile().score / 100.0
            except:
                safety_score = 0.0
            
            relevance_score = self._calculate_prompt_relevance(molecule, prompt)
            
            # Novelty based on structure complexity
            smiles = molecule.smiles
            novelty_score = len(set(smiles)) / len(smiles) if smiles else 0.0
            
            return (validity_score + safety_score + relevance_score + novelty_score) / 4.0
        
        return fitness_function
    
    def _update_learning_metrics(self, quality_progression: List[Dict[str, float]],
                               optimization_actions: List[str]):
        """Update learning metrics based on optimization results."""
        
        if not quality_progression:
            return
        
        # Quality trend
        overall_scores = [q.get('overall_score', 0.0) for q in quality_progression]
        self.learning_metrics.generation_quality_trend = overall_scores
        
        # Prompt understanding (based on relevance improvements)
        relevance_scores = [q.get('relevance_score', 0.0) for q in quality_progression]
        if len(relevance_scores) > 1:
            relevance_improvement = relevance_scores[-1] - relevance_scores[0]
            self.learning_metrics.prompt_understanding_score = max(0.0, min(1.0, 0.5 + relevance_improvement))
        
        # Optimization effectiveness
        if len(overall_scores) > 1:
            improvement = overall_scores[-1] - overall_scores[0]
            self.learning_metrics.optimization_effectiveness = max(0.0, min(1.0, 0.5 + improvement))
        
        # Convergence rate
        if len(overall_scores) >= 3:
            # Simple convergence measure
            recent_variance = np.std(overall_scores[-3:])
            self.learning_metrics.convergence_rate = max(0.0, 1.0 - recent_variance)
        
        # Exploration-exploitation ratio
        unique_actions = len(set(optimization_actions))
        total_actions = len(optimization_actions)
        self.learning_metrics.exploration_exploitation_ratio = unique_actions / max(total_actions, 1)
        
        # Learning stability
        if len(overall_scores) >= 2:
            stability = 1.0 - np.std(overall_scores) / max(np.mean(overall_scores), 0.01)
            self.learning_metrics.learning_stability = max(0.0, min(1.0, stability))
    
    def _calculate_improvement(self, quality_progression: List[Dict[str, float]]) -> float:
        """Calculate overall improvement from optimization."""
        
        if len(quality_progression) < 2:
            return 0.0
        
        initial_score = quality_progression[0].get('overall_score', 0.0)
        final_score = quality_progression[-1].get('overall_score', 0.0)
        
        return final_score - initial_score
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations based on learning results."""
        
        recommendations = []
        
        # Quality trend analysis
        if len(self.learning_metrics.generation_quality_trend) >= 2:
            trend = self.learning_metrics.generation_quality_trend[-1] - self.learning_metrics.generation_quality_trend[0]
            
            if trend > 0.1:
                recommendations.append("Optimization is showing strong positive results. Continue current approach.")
            elif trend > 0:
                recommendations.append("Optimization showing modest improvement. Consider more aggressive strategies.")
            else:
                recommendations.append("Optimization not improving quality. Consider alternative approaches.")
        
        # Learning effectiveness
        if self.learning_metrics.optimization_effectiveness < 0.3:
            recommendations.append("Low optimization effectiveness. Increase exploration rate or try different actions.")
        
        # Convergence analysis
        if self.learning_metrics.convergence_rate < 0.5:
            recommendations.append("Poor convergence. Consider adjusting learning parameters or increasing iterations.")
        
        # Exploration-exploitation balance
        if self.learning_metrics.exploration_exploitation_ratio < 0.3:
            recommendations.append("Low exploration. Increase epsilon or try more diverse optimization actions.")
        elif self.learning_metrics.exploration_exploitation_ratio > 0.8:
            recommendations.append("High exploration. Consider more exploitation of successful strategies.")
        
        if not recommendations:
            recommendations.append("Optimization performance is balanced. Continue with current configuration.")
        
        return recommendations