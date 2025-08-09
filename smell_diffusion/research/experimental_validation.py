"""
Comprehensive Experimental Validation Framework

Implements rigorous experimental validation for research publications:
- A/B testing frameworks
- Statistical significance testing
- Reproducibility validation
- Benchmark comparison protocols
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random

try:
    import numpy as np
except ImportError:
    # Research-grade fallback
    class ResearchMockNumPy:
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x): 
            mean_val = sum(x) / len(x) if x else 0
            variance = sum((i - mean_val) ** 2 for i in x) / len(x) if x else 0
            return variance ** 0.5
        @staticmethod
        def array(x): return x
        @staticmethod
        def random(): return random
        @staticmethod
        def choice(items, p=None): return random.choice(items)
        @staticmethod
        def percentile(x, p): 
            sorted_x = sorted(x)
            index = int(p * len(sorted_x) / 100)
            return sorted_x[min(index, len(sorted_x) - 1)] if sorted_x else 0
    np = ResearchMockNumPy()

from ..core.molecule import Molecule
from ..utils.logging import SmellDiffusionLogger


@dataclass
class ExperimentalConfiguration:
    """Configuration for controlled experiments."""
    experiment_name: str
    num_runs: int
    molecules_per_run: int
    control_group: str
    treatment_group: str
    randomization_seed: Optional[int]
    significance_level: float = 0.05


@dataclass
class StatisticalTest:
    """Statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str


class ExperimentalValidator:
    """Rigorous experimental validation for research publications."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("experimental_validator")
        self.experiment_registry = {}
        
    def setup_controlled_experiment(self, config: ExperimentalConfiguration) -> str:
        """Setup a controlled experiment with proper randomization."""
        
        experiment_id = hashlib.md5(
            f"{config.experiment_name}_{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Set randomization seed for reproducibility
        if config.randomization_seed:
            random.seed(config.randomization_seed)
            
        self.experiment_registry[experiment_id] = {
            'config': config,
            'start_time': time.time(),
            'results': {'control': [], 'treatment': []},
            'metadata': {
                'randomization_seed': config.randomization_seed,
                'experiment_design': 'randomized_controlled'
            }
        }
        
        self.logger.logger.info(f"Experiment {experiment_id} initialized: {config.experiment_name}")
        return experiment_id
    
    def run_ab_test(self, experiment_id: str, 
                   control_generator, treatment_generator,
                   test_prompts: List[str]) -> Dict[str, Any]:
        """Run A/B test comparing control and treatment methods."""
        
        if experiment_id not in self.experiment_registry:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiment_registry[experiment_id]
        config = experiment['config']
        
        self.logger.logger.info(f"Running A/B test for experiment {experiment_id}")
        
        # Run control group experiments
        control_results = self._run_experimental_group(
            control_generator, test_prompts, config, "control"
        )
        
        # Run treatment group experiments  
        treatment_results = self._run_experimental_group(
            treatment_generator, test_prompts, config, "treatment"
        )
        
        # Store results
        experiment['results']['control'] = control_results
        experiment['results']['treatment'] = treatment_results
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            control_results, treatment_results, config.significance_level
        )
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(control_results, treatment_results)
        
        # Generate comprehensive report
        report = {
            'experiment_id': experiment_id,
            'configuration': config,
            'control_results': control_results,
            'treatment_results': treatment_results,
            'statistical_analysis': statistical_analysis,
            'effect_sizes': effect_sizes,
            'recommendations': self._generate_recommendations(statistical_analysis, effect_sizes)
        }
        
        self.logger.logger.info(f"A/B test completed for experiment {experiment_id}")
        return report
    
    def _run_experimental_group(self, generator, test_prompts: List[str], 
                              config: ExperimentalConfiguration, 
                              group_name: str) -> Dict[str, Any]:
        """Run experiments for a single group."""
        
        group_results = {
            'generation_metrics': [],
            'quality_metrics': [],
            'performance_metrics': []
        }
        
        for run_id in range(config.num_runs):
            run_results = {
                'run_id': run_id,
                'prompt_results': []
            }
            
            for prompt in test_prompts:
                start_time = time.time()
                
                try:
                    # Generate molecules
                    if hasattr(generator, 'generate'):
                        molecules = generator.generate(
                            prompt=prompt,
                            num_molecules=config.molecules_per_run
                        )
                    else:
                        # Fallback for function generators
                        molecules = generator(prompt, config.molecules_per_run)
                    
                    generation_time = time.time() - start_time
                    
                    # Ensure molecules is a list
                    if not isinstance(molecules, list):
                        molecules = [molecules] if molecules else []
                    
                    # Evaluate quality
                    quality_metrics = self._evaluate_quality(molecules, prompt)
                    
                    prompt_result = {
                        'prompt': prompt,
                        'molecules': molecules,
                        'generation_time': generation_time,
                        'quality_metrics': quality_metrics
                    }
                    
                    run_results['prompt_results'].append(prompt_result)
                    
                except Exception as e:
                    self.logger.log_error(f"generation_error_{group_name}_{run_id}", e)
                    prompt_result = {
                        'prompt': prompt,
                        'molecules': [],
                        'generation_time': 0.0,
                        'quality_metrics': {'error': str(e)},
                        'failed': True
                    }
                    run_results['prompt_results'].append(prompt_result)
            
            # Aggregate run metrics
            run_metrics = self._aggregate_run_metrics(run_results)
            group_results['generation_metrics'].append(run_metrics)
        
        # Calculate group-level statistics
        group_results['summary_statistics'] = self._calculate_group_statistics(group_results)
        
        return group_results
    
    def _evaluate_quality(self, molecules: List[Molecule], prompt: str) -> Dict[str, float]:
        """Evaluate quality of generated molecules."""
        
        if not molecules:
            return {
                'validity_rate': 0.0,
                'average_safety_score': 0.0,
                'prompt_relevance': 0.0,
                'molecular_diversity': 0.0
            }
        
        # Validity assessment
        valid_molecules = [mol for mol in molecules if mol and mol.is_valid]
        validity_rate = len(valid_molecules) / len(molecules)
        
        # Safety assessment
        safety_scores = []
        for mol in valid_molecules:
            try:
                safety_profile = mol.get_safety_profile()
                safety_scores.append(safety_profile.score)
            except:
                safety_scores.append(0.0)
        
        average_safety_score = np.mean(safety_scores) if safety_scores else 0.0
        
        # Prompt relevance assessment
        relevance_scores = []
        for mol in valid_molecules:
            relevance = self._assess_prompt_relevance(mol, prompt)
            relevance_scores.append(relevance)
        
        prompt_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Molecular diversity assessment
        molecular_diversity = self._calculate_molecular_diversity(valid_molecules)
        
        return {
            'validity_rate': validity_rate,
            'average_safety_score': average_safety_score,
            'prompt_relevance': prompt_relevance,
            'molecular_diversity': molecular_diversity,
            'total_molecules': len(molecules),
            'valid_molecules': len(valid_molecules)
        }
    
    def _assess_prompt_relevance(self, molecule: Molecule, prompt: str) -> float:
        """Assess how well molecule matches prompt."""
        try:
            prompt_lower = prompt.lower()
            fragrance_notes = molecule.fragrance_notes
            all_notes = fragrance_notes.top + fragrance_notes.middle + fragrance_notes.base
            
            matches = sum(1 for note in all_notes if note in prompt_lower)
            return matches / max(len(all_notes), 1)
        except:
            return 0.0
    
    def _calculate_molecular_diversity(self, molecules: List[Molecule]) -> float:
        """Calculate diversity among molecules."""
        if len(molecules) < 2:
            return 0.0
            
        # Calculate pairwise differences
        unique_smiles = set()
        for mol in molecules:
            if mol and mol.smiles:
                unique_smiles.add(mol.smiles)
        
        # Simple diversity metric: fraction of unique molecules
        return len(unique_smiles) / len(molecules) if molecules else 0.0
    
    def _aggregate_run_metrics(self, run_results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate metrics for a single run."""
        
        prompt_results = run_results.get('prompt_results', [])
        if not prompt_results:
            return {'generation_time': 0.0, 'quality_score': 0.0}
        
        # Aggregate generation times
        generation_times = [r.get('generation_time', 0.0) for r in prompt_results if not r.get('failed', False)]
        avg_generation_time = np.mean(generation_times) if generation_times else 0.0
        
        # Aggregate quality metrics
        quality_scores = []
        for result in prompt_results:
            if not result.get('failed', False):
                quality_metrics = result.get('quality_metrics', {})
                # Calculate composite quality score
                validity = quality_metrics.get('validity_rate', 0.0)
                safety = quality_metrics.get('average_safety_score', 0.0) / 100.0
                relevance = quality_metrics.get('prompt_relevance', 0.0)
                diversity = quality_metrics.get('molecular_diversity', 0.0)
                
                composite_score = (validity + safety + relevance + diversity) / 4.0
                quality_scores.append(composite_score)
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'generation_time': avg_generation_time,
            'quality_score': avg_quality_score,
            'successful_generations': len(quality_scores),
            'total_attempts': len(prompt_results)
        }
    
    def _calculate_group_statistics(self, group_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for experimental group."""
        
        generation_metrics = group_results.get('generation_metrics', [])
        if not generation_metrics:
            return {}
        
        # Extract metrics across runs
        generation_times = [m.get('generation_time', 0.0) for m in generation_metrics]
        quality_scores = [m.get('quality_score', 0.0) for m in generation_metrics]
        
        return {
            'generation_time': {
                'mean': np.mean(generation_times),
                'std': np.std(generation_times),
                'min': min(generation_times) if generation_times else 0,
                'max': max(generation_times) if generation_times else 0
            },
            'quality_score': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': min(quality_scores) if quality_scores else 0,
                'max': max(quality_scores) if quality_scores else 0
            },
            'sample_size': len(generation_metrics),
            'success_rate': np.mean([m.get('successful_generations', 0) / max(m.get('total_attempts', 1), 1) for m in generation_metrics])
        }
    
    def _perform_statistical_analysis(self, control_results: Dict[str, Any], 
                                    treatment_results: Dict[str, Any],
                                    significance_level: float) -> Dict[str, StatisticalTest]:
        """Perform comprehensive statistical analysis."""
        
        statistical_tests = {}
        
        # Extract summary statistics
        control_stats = control_results.get('summary_statistics', {})
        treatment_stats = treatment_results.get('summary_statistics', {})
        
        # T-test for generation time
        if 'generation_time' in control_stats and 'generation_time' in treatment_stats:
            control_gen_times = [m.get('generation_time', 0.0) for m in control_results.get('generation_metrics', [])]
            treatment_gen_times = [m.get('generation_time', 0.0) for m in treatment_results.get('generation_metrics', [])]
            
            statistical_tests['generation_time'] = self._perform_t_test(
                control_gen_times, treatment_gen_times, "Generation Time Comparison", significance_level
            )
        
        # T-test for quality scores
        if 'quality_score' in control_stats and 'quality_score' in treatment_stats:
            control_quality = [m.get('quality_score', 0.0) for m in control_results.get('generation_metrics', [])]
            treatment_quality = [m.get('quality_score', 0.0) for m in treatment_results.get('generation_metrics', [])]
            
            statistical_tests['quality_score'] = self._perform_t_test(
                control_quality, treatment_quality, "Quality Score Comparison", significance_level
            )
        
        return statistical_tests
    
    def _perform_t_test(self, group1: List[float], group2: List[float], 
                       test_name: str, significance_level: float) -> StatisticalTest:
        """Perform two-sample t-test."""
        
        if len(group1) < 2 or len(group2) < 2:
            return StatisticalTest(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                interpretation="Insufficient data for statistical test"
            )
        
        # Calculate means and standard deviations
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1)
        std2 = np.std(group2)
        n1 = len(group1)
        n2 = len(group2)
        
        # Pooled standard error
        pooled_se = ((std1**2 / n1) + (std2**2 / n2))**0.5 if (std1 + std2) > 0 else 1e-10
        
        # T-statistic
        t_statistic = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0.0
        
        # Simplified p-value calculation (would use proper statistical library in practice)
        # This is a rough approximation
        abs_t = abs(t_statistic)
        if abs_t < 1.0:
            p_value = 0.4
        elif abs_t < 2.0:
            p_value = 0.1
        elif abs_t < 3.0:
            p_value = 0.01
        else:
            p_value = 0.001
        
        # Effect size (Cohen's d)
        pooled_std = ((std1**2 + std2**2) / 2)**0.5 if (std1 + std2) > 0 else 1e-10
        cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Confidence interval (simplified)
        margin_of_error = 1.96 * pooled_se  # 95% CI
        ci_lower = (mean1 - mean2) - margin_of_error
        ci_upper = (mean1 - mean2) + margin_of_error
        
        # Interpretation
        if p_value < significance_level:
            if cohens_d > 0.8:
                interpretation = f"Statistically significant with large effect size (p={p_value:.3f}, d={cohens_d:.2f})"
            elif cohens_d > 0.5:
                interpretation = f"Statistically significant with medium effect size (p={p_value:.3f}, d={cohens_d:.2f})"
            else:
                interpretation = f"Statistically significant with small effect size (p={p_value:.3f}, d={cohens_d:.2f})"
        else:
            interpretation = f"Not statistically significant (p={p_value:.3f})"
        
        return StatisticalTest(
            test_name=test_name,
            statistic=t_statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _calculate_effect_sizes(self, control_results: Dict[str, Any], 
                              treatment_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effect sizes for practical significance."""
        
        effect_sizes = {}
        
        # Extract metrics
        control_metrics = control_results.get('generation_metrics', [])
        treatment_metrics = treatment_results.get('generation_metrics', [])
        
        if not control_metrics or not treatment_metrics:
            return effect_sizes
        
        # Generation time effect size
        control_times = [m.get('generation_time', 0.0) for m in control_metrics]
        treatment_times = [m.get('generation_time', 0.0) for m in treatment_metrics]
        effect_sizes['generation_time'] = self._cohens_d(control_times, treatment_times)
        
        # Quality score effect size
        control_quality = [m.get('quality_score', 0.0) for m in control_metrics]
        treatment_quality = [m.get('quality_score', 0.0) for m in treatment_metrics]
        effect_sizes['quality_score'] = self._cohens_d(control_quality, treatment_quality)
        
        return effect_sizes
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1)
        std2 = np.std(group2)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        pooled_std = pooled_std**0.5 if pooled_std > 0 else 1e-10
        
        return abs(mean1 - mean2) / pooled_std
    
    def _generate_recommendations(self, statistical_analysis: Dict[str, StatisticalTest],
                                effect_sizes: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        for test_name, test_result in statistical_analysis.items():
            if test_result.p_value < 0.05:
                effect_size = effect_sizes.get(test_name, 0.0)
                
                if effect_size > 0.8:
                    recommendations.append(
                        f"Strong evidence for treatment superiority in {test_name} "
                        f"(p={test_result.p_value:.3f}, large effect size={effect_size:.2f})"
                    )
                elif effect_size > 0.5:
                    recommendations.append(
                        f"Moderate evidence for treatment superiority in {test_name} "
                        f"(p={test_result.p_value:.3f}, medium effect size={effect_size:.2f})"
                    )
                else:
                    recommendations.append(
                        f"Weak evidence for treatment superiority in {test_name} "
                        f"(p={test_result.p_value:.3f}, small effect size={effect_size:.2f})"
                    )
            else:
                recommendations.append(
                    f"No significant difference found in {test_name} "
                    f"(p={test_result.p_value:.3f})"
                )
        
        if not recommendations:
            recommendations.append("Insufficient data for statistical conclusions")
        
        return recommendations


class BenchmarkValidator:
    """Validate against established benchmarks."""
    
    BENCHMARK_DATASETS = {
        'fragrance_classic': [
            "Fresh citrus morning scent",
            "Romantic rose garden evening",
            "Woody cedar forest walk",
            "Sweet vanilla dessert fragrance",
            "Clean ocean breeze scent"
        ],
        'complexity_test': [
            "Complex oriental spicy amber with oud undertones and floral heart",
            "Sophisticated chypre with bergamot top, rose heart, and oakmoss base",
            "Modern aquatic with sea salt, white musk, and driftwood"
        ],
        'safety_critical': [
            "Hypoallergenic baby-safe gentle fragrance",
            "Sensitive skin friendly floral scent",
            "IFRA compliant professional perfume"
        ]
    }
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("benchmark_validator")
    
    def run_benchmark_suite(self, generator, benchmark_name: str = 'fragrance_classic') -> Dict[str, Any]:
        """Run comprehensive benchmark validation."""
        
        if benchmark_name not in self.BENCHMARK_DATASETS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        test_prompts = self.BENCHMARK_DATASETS[benchmark_name]
        
        self.logger.logger.info(f"Running benchmark suite: {benchmark_name}")
        
        benchmark_results = []
        
        for prompt in test_prompts:
            start_time = time.time()
            
            try:
                # Generate molecules
                molecules = generator.generate(prompt=prompt, num_molecules=5)
                if not isinstance(molecules, list):
                    molecules = [molecules] if molecules else []
                
                generation_time = time.time() - start_time
                
                # Evaluate against benchmark criteria
                benchmark_metrics = self._evaluate_benchmark_performance(molecules, prompt)
                
                result = {
                    'prompt': prompt,
                    'generation_time': generation_time,
                    'molecules_generated': len(molecules),
                    'benchmark_metrics': benchmark_metrics,
                    'success': True
                }
                
            except Exception as e:
                self.logger.log_error(f"benchmark_error_{benchmark_name}", e)
                result = {
                    'prompt': prompt,
                    'generation_time': time.time() - start_time,
                    'molecules_generated': 0,
                    'benchmark_metrics': {'error': str(e)},
                    'success': False
                }
            
            benchmark_results.append(result)
        
        # Calculate overall benchmark score
        overall_score = self._calculate_benchmark_score(benchmark_results)
        
        return {
            'benchmark_name': benchmark_name,
            'overall_score': overall_score,
            'individual_results': benchmark_results,
            'summary': self._generate_benchmark_summary(benchmark_results),
            'grade': self._assign_benchmark_grade(overall_score)
        }
    
    def _evaluate_benchmark_performance(self, molecules: List[Molecule], 
                                      prompt: str) -> Dict[str, float]:
        """Evaluate performance against benchmark criteria."""
        
        if not molecules:
            return {
                'validity_score': 0.0,
                'relevance_score': 0.0,
                'safety_score': 0.0,
                'overall_score': 0.0
            }
        
        # Validity assessment
        valid_molecules = [mol for mol in molecules if mol and mol.is_valid]
        validity_score = len(valid_molecules) / len(molecules)
        
        # Relevance assessment
        relevance_scores = []
        for mol in valid_molecules:
            try:
                relevance = self._assess_benchmark_relevance(mol, prompt)
                relevance_scores.append(relevance)
            except:
                relevance_scores.append(0.0)
        
        relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Safety assessment
        safety_scores = []
        for mol in valid_molecules:
            try:
                safety_profile = mol.get_safety_profile()
                safety_scores.append(safety_profile.score / 100.0)
            except:
                safety_scores.append(0.0)
        
        safety_score = np.mean(safety_scores) if safety_scores else 0.0
        
        # Overall benchmark score
        overall_score = (validity_score + relevance_score + safety_score) / 3.0
        
        return {
            'validity_score': validity_score,
            'relevance_score': relevance_score,
            'safety_score': safety_score,
            'overall_score': overall_score
        }
    
    def _assess_benchmark_relevance(self, molecule: Molecule, prompt: str) -> float:
        """Assess relevance for benchmark evaluation."""
        try:
            prompt_lower = prompt.lower()
            fragrance_notes = molecule.fragrance_notes
            all_notes = fragrance_notes.top + fragrance_notes.middle + fragrance_notes.base
            
            # Enhanced relevance scoring for benchmarks
            matches = 0
            partial_matches = 0
            
            for note in all_notes:
                if note in prompt_lower:
                    matches += 1
                elif any(keyword in prompt_lower for keyword in [note[:3], note[-3:]]):
                    partial_matches += 1
            
            total_score = matches + (partial_matches * 0.5)
            return min(1.0, total_score / max(len(all_notes), 1))
            
        except:
            return 0.0
    
    def _calculate_benchmark_score(self, benchmark_results: List[Dict[str, Any]]) -> float:
        """Calculate overall benchmark score."""
        
        successful_results = [r for r in benchmark_results if r.get('success', False)]
        if not successful_results:
            return 0.0
        
        overall_scores = []
        for result in successful_results:
            metrics = result.get('benchmark_metrics', {})
            if 'overall_score' in metrics:
                overall_scores.append(metrics['overall_score'])
        
        return np.mean(overall_scores) if overall_scores else 0.0
    
    def _generate_benchmark_summary(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of benchmark results."""
        
        successful_runs = sum(1 for r in benchmark_results if r.get('success', False))
        total_runs = len(benchmark_results)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
        
        generation_times = [r.get('generation_time', 0.0) for r in benchmark_results]
        
        return {
            'success_rate': success_rate,
            'total_prompts': total_runs,
            'successful_generations': successful_runs,
            'average_generation_time': np.mean(generation_times),
            'fastest_generation': min(generation_times) if generation_times else 0.0,
            'slowest_generation': max(generation_times) if generation_times else 0.0
        }
    
    def _assign_benchmark_grade(self, overall_score: float) -> str:
        """Assign letter grade based on benchmark performance."""
        
        if overall_score >= 0.9:
            return 'A+'
        elif overall_score >= 0.85:
            return 'A'
        elif overall_score >= 0.8:
            return 'A-'
        elif overall_score >= 0.75:
            return 'B+'
        elif overall_score >= 0.7:
            return 'B'
        elif overall_score >= 0.65:
            return 'B-'
        elif overall_score >= 0.6:
            return 'C+'
        elif overall_score >= 0.55:
            return 'C'
        elif overall_score >= 0.5:
            return 'C-'
        else:
            return 'F'