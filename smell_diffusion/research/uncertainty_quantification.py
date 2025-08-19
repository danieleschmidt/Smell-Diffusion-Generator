"""
Uncertainty Quantification for Molecular Generation

Implementation of Bayesian neural networks and ensemble methods to provide
uncertainty estimates for generated molecules, improving reliability and trust.

Research Hypothesis: Bayesian neural networks and ensemble methods will improve 
reliability by providing uncertainty estimates for generated molecules with 95% calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class UncertaintyPrediction:
    """Container for predictions with uncertainty estimates"""
    mean_prediction: torch.Tensor
    epistemic_uncertainty: torch.Tensor  # Model uncertainty
    aleatoric_uncertainty: torch.Tensor  # Data uncertainty
    total_uncertainty: torch.Tensor
    confidence_interval: Tuple[torch.Tensor, torch.Tensor]
    prediction_interval: Tuple[torch.Tensor, torch.Tensor]

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -3.0))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -3.0))
        
        # Prior distributions
        self.weight_prior_std = prior_std
        self.bias_prior_std = prior_std
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with reparameterization trick"""
        if sample and self.training:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_eps * weight_std
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_eps * bias_std
            
        else:
            # Use mean weights (deterministic)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # Weight KL divergence
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_var) / (self.weight_prior_std ** 2) -
            self.weight_logvar + torch.log(torch.tensor(self.weight_prior_std ** 2)) - 1
        )
        
        # Bias KL divergence
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_var) / (self.bias_prior_std ** 2) -
            self.bias_logvar + torch.log(torch.tensor(self.bias_prior_std ** 2)) - 1
        )
        
        return weight_kl + bias_kl

class BayesianMolecularPredictor(nn.Module):
    """Bayesian neural network for molecular property prediction with uncertainty"""
    
    def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = [512, 256, 128], 
                 output_dim: int = 1, num_classes: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.is_classification = num_classes is not None
        
        # Build Bayesian layers
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(BayesianLinear(dims[i], dims[i + 1]))
        
        # Output layer
        output_size = num_classes if self.is_classification else output_dim
        self.output_layer = BayesianLinear(hidden_dims[-1], output_size)
        
        # Aleatoric uncertainty parameter (for regression)
        if not self.is_classification:
            self.log_noise_var = nn.Parameter(torch.tensor(-2.0))
        
        # Dropout for additional uncertainty
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty quantification"""
        if num_samples == 1 and not self.training:
            # Single deterministic forward pass
            h = x
            for layer in self.layers:
                h = F.relu(layer(h, sample=False))
                h = self.dropout(h)
            
            output = self.output_layer(h, sample=False)
            
            if self.is_classification:
                logits = output
                probs = F.softmax(logits, dim=-1)
                return {
                    'predictions': probs,
                    'logits': logits,
                    'epistemic_uncertainty': torch.zeros_like(probs),
                    'total_uncertainty': torch.zeros_like(probs)
                }
            else:
                mean_pred = output
                noise_var = torch.exp(self.log_noise_var)
                return {
                    'predictions': mean_pred,
                    'aleatoric_uncertainty': noise_var.expand_as(mean_pred),
                    'epistemic_uncertainty': torch.zeros_like(mean_pred),
                    'total_uncertainty': noise_var.expand_as(mean_pred)
                }
        
        # Monte Carlo sampling for uncertainty estimation
        predictions = []
        kl_divergences = []
        
        for _ in range(num_samples):
            h = x
            layer_kls = []
            
            for layer in self.layers:
                h = F.relu(layer(h, sample=True))
                h = self.dropout(h)
                layer_kls.append(layer.kl_divergence())
            
            output = self.output_layer(h, sample=True)
            layer_kls.append(self.output_layer.kl_divergence())
            
            if self.is_classification:
                probs = F.softmax(output, dim=-1)
                predictions.append(probs)
            else:
                predictions.append(output)
            
            kl_divergences.append(sum(layer_kls))
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        # Compute uncertainty measures
        mean_pred = predictions.mean(dim=0)
        
        if self.is_classification:
            # For classification: predictive entropy and mutual information
            epistemic_uncertainty = self._compute_epistemic_uncertainty_classification(predictions)
            total_uncertainty = self._compute_total_uncertainty_classification(predictions)
            
            return {
                'predictions': mean_pred,
                'samples': predictions,
                'epistemic_uncertainty': epistemic_uncertainty,
                'total_uncertainty': total_uncertainty,
                'kl_divergence': torch.stack(kl_divergences).mean()
            }
        else:
            # For regression: epistemic and aleatoric uncertainty
            epistemic_uncertainty = predictions.var(dim=0)  # Variance across samples
            noise_var = torch.exp(self.log_noise_var)
            aleatoric_uncertainty = noise_var.expand_as(mean_pred)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            return {
                'predictions': mean_pred,
                'samples': predictions,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric_uncertainty,
                'total_uncertainty': total_uncertainty,
                'kl_divergence': torch.stack(kl_divergences).mean()
            }
    
    def _compute_epistemic_uncertainty_classification(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty for classification (mutual information)"""
        mean_probs = predictions.mean(dim=0)
        entropy_of_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        sample_entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1)
        mean_of_entropies = sample_entropies.mean(dim=0)
        
        mutual_information = entropy_of_mean - mean_of_entropies
        return mutual_information
    
    def _compute_total_uncertainty_classification(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute total uncertainty for classification (predictive entropy)"""
        mean_probs = predictions.mean(dim=0)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        return predictive_entropy

class EnsembleMolecularPredictor(nn.Module):
    """Ensemble of models for uncertainty quantification"""
    
    def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 1, num_models: int = 5, num_classes: Optional[int] = None):
        super().__init__()
        self.num_models = num_models
        self.is_classification = num_classes is not None
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            BayesianMolecularPredictor(input_dim, hidden_dims, output_dim, num_classes)
            for _ in range(num_models)
        ])
        
    def forward(self, x: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble with uncertainty quantification"""
        ensemble_predictions = []
        ensemble_uncertainties = []
        
        for model in self.models:
            model_output = model(x, num_samples=num_samples)
            ensemble_predictions.append(model_output['predictions'])
            
            if 'total_uncertainty' in model_output:
                ensemble_uncertainties.append(model_output['total_uncertainty'])
        
        # Stack ensemble predictions
        ensemble_predictions = torch.stack(ensemble_predictions, dim=0)
        
        # Compute ensemble statistics
        mean_prediction = ensemble_predictions.mean(dim=0)
        
        # Epistemic uncertainty: variance across ensemble members
        epistemic_uncertainty = ensemble_predictions.var(dim=0)
        
        # Average aleatoric uncertainty from individual models
        if ensemble_uncertainties:
            aleatoric_uncertainty = torch.stack(ensemble_uncertainties, dim=0).mean(dim=0)
        else:
            aleatoric_uncertainty = torch.zeros_like(mean_prediction)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': mean_prediction,
            'ensemble_predictions': ensemble_predictions,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty
        }

class UncertaintyCalibration:
    """Tools for calibrating and evaluating uncertainty estimates"""
    
    def __init__(self):
        self.calibration_curves = {}
        
    def compute_calibration_curve(self, confidences: np.ndarray, accuracies: np.ndarray,
                                num_bins: int = 10) -> Dict[str, np.ndarray]:
        """Compute reliability diagram for calibration assessment"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        return {
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries
        }
    
    def expected_calibration_error(self, confidences: np.ndarray, accuracies: np.ndarray,
                                 num_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)"""
        calibration_curve = self.compute_calibration_curve(confidences, accuracies, num_bins)
        
        bin_counts = calibration_curve['bin_counts']
        bin_accuracies = calibration_curve['bin_accuracies']
        bin_confidences = calibration_curve['bin_confidences']
        
        total_samples = bin_counts.sum()
        ece = 0
        
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences):
            if count > 0:
                ece += (count / total_samples) * abs(acc - conf)
        
        return ece
    
    def plot_calibration_curve(self, confidences: np.ndarray, accuracies: np.ndarray,
                             title: str = "Calibration Curve", save_path: Optional[str] = None):
        """Plot reliability diagram"""
        calibration_curve = self.compute_calibration_curve(confidences, accuracies)
        
        plt.figure(figsize=(8, 6))
        
        # Plot calibration curve
        plt.plot(calibration_curve['bin_confidences'], calibration_curve['bin_accuracies'],
                'o-', linewidth=2, markersize=8, label='Model')
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
        
        # Add bar chart for sample counts
        ax2 = plt.gca().twinx()
        ax2.bar(calibration_curve['bin_confidences'], calibration_curve['bin_counts'],
                alpha=0.3, width=0.08, color='lightblue', label='Sample Count')
        ax2.set_ylabel('Sample Count')
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Compute and display ECE
        ece = self.expected_calibration_error(confidences, accuracies)
        plt.text(0.02, 0.98, f'ECE: {ece:.3f}', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class UncertaintyAwareMolecularGenerator:
    """Molecular generator with uncertainty-aware sampling and filtering"""
    
    def __init__(self, base_generator, uncertainty_predictor: Union[BayesianMolecularPredictor, EnsembleMolecularPredictor]):
        self.base_generator = base_generator
        self.uncertainty_predictor = uncertainty_predictor
        self.calibration = UncertaintyCalibration()
        
    def generate_with_uncertainty(self, prompt: str, num_molecules: int = 10, 
                                uncertainty_threshold: float = 0.1,
                                confidence_level: float = 0.95) -> Dict[str, any]:
        """Generate molecules with uncertainty filtering"""
        logger.info(f"Generating {num_molecules} molecules with uncertainty threshold {uncertainty_threshold}")
        
        # Generate initial set of molecules (oversample to account for filtering)
        oversample_factor = 3
        initial_molecules = self.base_generator.generate(
            prompt=prompt,
            num_molecules=num_molecules * oversample_factor,
            return_features=True
        )
        
        # Extract molecular features for uncertainty prediction
        molecular_features = self._extract_molecular_features(initial_molecules)
        
        # Predict properties with uncertainty
        with torch.no_grad():
            uncertainty_output = self.uncertainty_predictor(molecular_features, num_samples=50)
        
        predictions = uncertainty_output['predictions']
        uncertainties = uncertainty_output['total_uncertainty']
        
        # Filter molecules based on uncertainty threshold
        low_uncertainty_mask = uncertainties.squeeze() < uncertainty_threshold
        filtered_indices = torch.where(low_uncertainty_mask)[0]
        
        if len(filtered_indices) < num_molecules:
            logger.warning(f"Only {len(filtered_indices)} molecules meet uncertainty threshold, "
                          f"returning all available molecules")
            selected_indices = filtered_indices
        else:
            # Select top molecules by prediction confidence
            confidence_scores = 1 / (1 + uncertainties.squeeze())
            selected_indices = filtered_indices[
                torch.topk(confidence_scores[filtered_indices], num_molecules).indices
            ]
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            uncertainty_output, confidence_level, selected_indices
        )
        
        # Prepare results
        selected_molecules = [initial_molecules[i] for i in selected_indices.cpu().numpy()]
        selected_predictions = predictions[selected_indices]
        selected_uncertainties = uncertainties[selected_indices]
        
        return {
            'molecules': selected_molecules,
            'predictions': selected_predictions,
            'uncertainties': selected_uncertainties,
            'confidence_intervals': confidence_intervals,
            'uncertainty_threshold': uncertainty_threshold,
            'filtering_rate': len(selected_indices) / len(initial_molecules),
            'total_generated': len(initial_molecules),
            'selected_count': len(selected_indices)
        }
    
    def _extract_molecular_features(self, molecules: List) -> torch.Tensor:
        """Extract features from molecules for uncertainty prediction"""
        # Placeholder for feature extraction
        # In practice, this would use molecular fingerprints, descriptors, etc.
        num_molecules = len(molecules)
        feature_dim = self.uncertainty_predictor.input_dim
        
        # Mock feature extraction
        features = torch.randn(num_molecules, feature_dim)
        return features
    
    def _compute_confidence_intervals(self, uncertainty_output: Dict, confidence_level: float,
                                    selected_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute confidence intervals for predictions"""
        predictions = uncertainty_output['predictions'][selected_indices]
        uncertainties = uncertainty_output['total_uncertainty'][selected_indices]
        
        # Assume Gaussian distribution for confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * torch.sqrt(uncertainties)
        
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin_of_error': margin_of_error,
            'confidence_level': confidence_level
        }
    
    def evaluate_uncertainty_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate the quality of uncertainty estimates"""
        all_predictions = []
        all_uncertainties = []
        all_true_values = []
        
        for data_point in test_data:
            features = self._extract_molecular_features([data_point['molecule']])
            
            with torch.no_grad():
                output = self.uncertainty_predictor(features, num_samples=50)
            
            predictions = output['predictions'].cpu().numpy()
            uncertainties = output['total_uncertainty'].cpu().numpy()
            true_values = np.array([data_point['true_value']])
            
            all_predictions.extend(predictions)
            all_uncertainties.extend(uncertainties)
            all_true_values.extend(true_values)
        
        all_predictions = np.array(all_predictions)
        all_uncertainties = np.array(all_uncertainties)
        all_true_values = np.array(all_true_values)
        
        # Compute metrics
        mae = np.mean(np.abs(all_predictions - all_true_values))
        rmse = np.sqrt(np.mean((all_predictions - all_true_values) ** 2))
        
        # Uncertainty quality metrics
        # Check if prediction errors correlate with uncertainties
        prediction_errors = np.abs(all_predictions - all_true_values)
        error_uncertainty_correlation = np.corrcoef(prediction_errors, all_uncertainties.flatten())[0, 1]
        
        # Compute calibration error for regression
        # Convert uncertainties to confidence scores
        confidences = 1 / (1 + all_uncertainties.flatten())
        # Binary accuracy: within 1 standard deviation
        within_bounds = prediction_errors <= all_uncertainties.flatten()
        
        ece = self.calibration.expected_calibration_error(confidences, within_bounds.astype(float))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'error_uncertainty_correlation': error_uncertainty_correlation,
            'expected_calibration_error': ece,
            'mean_uncertainty': np.mean(all_uncertainties),
            'uncertainty_std': np.std(all_uncertainties)
        }

# Experimental validation functions
def run_uncertainty_quantification_experiment() -> Dict[str, any]:
    """Run comprehensive uncertainty quantification experiment"""
    logger.info("Starting uncertainty quantification experiment")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Synthetic molecular features and properties
    num_samples = 1000
    feature_dim = 512
    
    X = torch.randn(num_samples, feature_dim)
    # True function with noise
    true_weights = torch.randn(feature_dim, 1) * 0.1
    noise_std = 0.2
    y_true = X @ true_weights + torch.randn(num_samples, 1) * noise_std
    
    # Split data
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_true[:split_idx], y_true[split_idx:]
    
    # Initialize models
    bayesian_model = BayesianMolecularPredictor(
        input_dim=feature_dim,
        hidden_dims=[256, 128],
        output_dim=1
    )
    
    ensemble_model = EnsembleMolecularPredictor(
        input_dim=feature_dim,
        hidden_dims=[256, 128],
        output_dim=1,
        num_models=5
    )
    
    # Training loop for Bayesian model
    optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=1e-3)
    bayesian_model.train()
    
    training_losses = []
    for epoch in range(200):
        optimizer.zero_grad()
        
        output = bayesian_model(X_train, num_samples=5)
        predictions = output['predictions']
        kl_div = output['kl_divergence']
        
        # Negative log-likelihood loss
        mse_loss = F.mse_loss(predictions, y_train)
        
        # Total loss with KL divergence
        kl_weight = 1.0 / len(X_train)  # Scale KL term
        total_loss = mse_loss + kl_weight * kl_div
        
        total_loss.backward()
        optimizer.step()
        
        training_losses.append(total_loss.item())
        
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}: Loss = {total_loss.item():.4f}, "
                       f"MSE = {mse_loss.item():.4f}, KL = {kl_div.item():.4f}")
    
    # Training ensemble models
    ensemble_losses = []
    for i, model in enumerate(ensemble_model.models):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(100):  # Fewer epochs per model
            optimizer.zero_grad()
            
            output = model(X_train, num_samples=3)
            predictions = output['predictions']
            kl_div = output['kl_divergence']
            
            mse_loss = F.mse_loss(predictions, y_train)
            kl_weight = 1.0 / len(X_train)
            total_loss = mse_loss + kl_weight * kl_div
            
            total_loss.backward()
            optimizer.step()
            
            ensemble_losses.append(total_loss.item())
    
    # Evaluation
    bayesian_model.eval()
    ensemble_model.eval()
    
    with torch.no_grad():
        # Bayesian model predictions
        bayesian_output = bayesian_model(X_test, num_samples=100)
        bayesian_preds = bayesian_output['predictions']
        bayesian_uncertainty = bayesian_output['total_uncertainty']
        
        # Ensemble model predictions  
        ensemble_output = ensemble_model(X_test, num_samples=20)
        ensemble_preds = ensemble_output['predictions']
        ensemble_uncertainty = ensemble_output['total_uncertainty']
    
    # Compute metrics
    bayesian_mae = F.l1_loss(bayesian_preds, y_test).item()
    bayesian_rmse = torch.sqrt(F.mse_loss(bayesian_preds, y_test)).item()
    
    ensemble_mae = F.l1_loss(ensemble_preds, y_test).item()
    ensemble_rmse = torch.sqrt(F.mse_loss(ensemble_preds, y_test)).item()
    
    # Uncertainty quality assessment
    bayesian_errors = torch.abs(bayesian_preds - y_test).cpu().numpy()
    bayesian_uncertainties = bayesian_uncertainty.cpu().numpy()
    
    ensemble_errors = torch.abs(ensemble_preds - y_test).cpu().numpy()
    ensemble_uncertainties = ensemble_uncertainty.cpu().numpy()
    
    # Error-uncertainty correlation
    bayesian_correlation = np.corrcoef(bayesian_errors.flatten(), bayesian_uncertainties.flatten())[0, 1]
    ensemble_correlation = np.corrcoef(ensemble_errors.flatten(), ensemble_uncertainties.flatten())[0, 1]
    
    results = {
        'bayesian_model': {
            'mae': bayesian_mae,
            'rmse': bayesian_rmse,
            'error_uncertainty_correlation': bayesian_correlation,
            'mean_uncertainty': np.mean(bayesian_uncertainties),
            'uncertainty_std': np.std(bayesian_uncertainties)
        },
        'ensemble_model': {
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'error_uncertainty_correlation': ensemble_correlation,
            'mean_uncertainty': np.mean(ensemble_uncertainties),
            'uncertainty_std': np.std(ensemble_uncertainties)
        },
        'training_losses': training_losses,
        'ensemble_losses': ensemble_losses
    }
    
    logger.info("Uncertainty quantification experiment completed")
    logger.info(f"Bayesian Model - MAE: {bayesian_mae:.4f}, RMSE: {bayesian_rmse:.4f}, "
               f"Error-Uncertainty Correlation: {bayesian_correlation:.4f}")
    logger.info(f"Ensemble Model - MAE: {ensemble_mae:.4f}, RMSE: {ensemble_rmse:.4f}, "
               f"Error-Uncertainty Correlation: {ensemble_correlation:.4f}")
    
    return results

if __name__ == "__main__":
    # Run experiment
    results = run_uncertainty_quantification_experiment()
    
    # Print summary
    print("\n=== Uncertainty Quantification Experiment Results ===")
    print(f"Bayesian Model Performance:")
    print(f"  MAE: {results['bayesian_model']['mae']:.4f}")
    print(f"  RMSE: {results['bayesian_model']['rmse']:.4f}")
    print(f"  Error-Uncertainty Correlation: {results['bayesian_model']['error_uncertainty_correlation']:.4f}")
    
    print(f"\nEnsemble Model Performance:")
    print(f"  MAE: {results['ensemble_model']['mae']:.4f}")
    print(f"  RMSE: {results['ensemble_model']['rmse']:.4f}")
    print(f"  Error-Uncertainty Correlation: {results['ensemble_model']['error_uncertainty_correlation']:.4f}")
    
    print("\nExperiment completed successfully!")