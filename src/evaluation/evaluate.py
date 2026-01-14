"""
Comprehensive model evaluation.

Implements all required metrics:
- AUC-ROC
- F1, Precision, Recall
- Specificity, Sensitivity
- Calibration (reliability diagrams)
- Confusion matrix
- Per-class metrics
- Threshold analysis
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import load_config, get_device, ensure_dir, set_seed
from src.data.dataset import create_dataloaders
from src.models.model import create_model


class Evaluator:
    """
    Comprehensive model evaluator.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        class_names: List of class names
        device: Device to run on
    """
    
    def __init__(self, model, test_loader, class_names, device):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
        
        # Results storage
        self.all_labels = []
        self.all_predictions = []
        self.all_probabilities = []
    
    def get_predictions(self):
        """Get predictions on test set."""
        print("\n" + "="*70)
        print("Getting Predictions on Test Set")
        print("="*70)
        
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                self.all_labels.extend(labels.cpu().numpy())
                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
        
        self.all_labels = np.array(self.all_labels)
        self.all_predictions = np.array(self.all_predictions)
        self.all_probabilities = np.array(self.all_probabilities)
        
        print(f"✓ Collected {len(self.all_labels)} predictions")
    
    def compute_metrics(self):
        """Compute all evaluation metrics."""
        print("\n" + "="*70)
        print("Computing Metrics")
        print("="*70)
        
        metrics = {}
        
        # Overall metrics
        accuracy = (self.all_predictions == self.all_labels).mean()
        metrics['accuracy'] = float(accuracy)
        
        # Per-class metrics
        precision = precision_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        recall = recall_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        f1 = f1_score(self.all_labels, self.all_predictions, average=None, zero_division=0)
        
        # Macro/micro averages
        precision_macro = precision_score(self.all_labels, self.all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(self.all_labels, self.all_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(self.all_labels, self.all_predictions, average='macro', zero_division=0)
        
        precision_micro = precision_score(self.all_labels, self.all_predictions, average='micro', zero_division=0)
        recall_micro = recall_score(self.all_labels, self.all_predictions, average='micro', zero_division=0)
        f1_micro = f1_score(self.all_labels, self.all_predictions, average='micro', zero_division=0)
        
        # Store
        metrics['precision_per_class'] = {self.class_names[i]: float(precision[i]) for i in range(self.num_classes)}
        metrics['recall_per_class'] = {self.class_names[i]: float(recall[i]) for i in range(self.num_classes)}
        metrics['f1_per_class'] = {self.class_names[i]: float(f1[i]) for i in range(self.num_classes)}
        metrics['sensitivity_per_class'] = metrics['recall_per_class']  # Sensitivity = Recall
        
        # Specificity per class
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        specificity = {}
        for i in range(self.num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity[self.class_names[i]] = float(spec)
        metrics['specificity_per_class'] = specificity
        
        # Averages
        metrics['precision_macro'] = float(precision_macro)
        metrics['recall_macro'] = float(recall_macro)
        metrics['f1_macro'] = float(f1_macro)
        metrics['sensitivity_macro'] = float(recall_macro)
        
        metrics['precision_micro'] = float(precision_micro)
        metrics['recall_micro'] = float(recall_micro)
        metrics['f1_micro'] = float(f1_micro)
        
        # AUC-ROC per class and macro
        auc_scores = {}
        for i in range(self.num_classes):
            # One-vs-rest
            binary_labels = (self.all_labels == i).astype(int)
            binary_probs = self.all_probabilities[:, i]
            try:
                auc_score = roc_auc_score(binary_labels, binary_probs)
                auc_scores[self.class_names[i]] = float(auc_score)
            except:
                auc_scores[self.class_names[i]] = 0.0
        
        metrics['auc_roc_per_class'] = auc_scores
        metrics['auc_roc_macro'] = float(np.mean(list(auc_scores.values())))
        
        # Print summary
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (micro): {f1_micro:.4f}")
        print(f"  AUC-ROC (macro): {metrics['auc_roc_macro']:.4f}")
        
        print(f"\n📋 Per-Class Metrics:")
        for class_name in self.class_names:
            print(f"\n  {class_name}:")
            print(f"    Precision:    {metrics['precision_per_class'][class_name]:.4f}")
            print(f"    Recall:       {metrics['recall_per_class'][class_name]:.4f}")
            print(f"    F1:           {metrics['f1_per_class'][class_name]:.4f}")
            print(f"    Sensitivity:  {metrics['sensitivity_per_class'][class_name]:.4f}")
            print(f"    Specificity:  {metrics['specificity_per_class'][class_name]:.4f}")
            print(f"    AUC-ROC:      {metrics['auc_roc_per_class'][class_name]:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path='reports/figures/confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, save_path='reports/figures/roc_curves.png'):
        """Plot ROC curves for each class."""
        fig, axes = plt.subplots(1, self.num_classes, figsize=(6*self.num_classes, 5))
        if self.num_classes == 1:
            axes = [axes]
        
        for i in range(self.num_classes):
            binary_labels = (self.all_labels == i).astype(int)
            binary_probs = self.all_probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(binary_labels, binary_probs)
            roc_auc = auc(fpr, tpr)
            
            axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate', fontsize=11)
            axes[i].set_ylabel('True Positive Rate', fontsize=11)
            axes[i].set_title(f'{self.class_names[i]}', fontsize=13, fontweight='bold')
            axes[i].legend(loc="lower right")
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to {save_path}")
    
    def plot_calibration(self, save_path='reports/figures/calibration.png'):
        """Plot reliability diagrams for calibration analysis."""
        fig, axes = plt.subplots(1, self.num_classes, figsize=(6*self.num_classes, 5))
        if self.num_classes == 1:
            axes = [axes]
        
        for i in range(self.num_classes):
            binary_labels = (self.all_labels == i).astype(int)
            binary_probs = self.all_probabilities[:, i]
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_labels, binary_probs, n_bins=10, strategy='uniform'
            )
            
            axes[i].plot(mean_predicted_value, fraction_of_positives, "s-",
                        label=f'{self.class_names[i]}', linewidth=2, markersize=8)
            axes[i].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=2)
            axes[i].set_xlabel('Mean Predicted Probability', fontsize=11)
            axes[i].set_ylabel('Fraction of Positives', fontsize=11)
            axes[i].set_title(f'Calibration: {self.class_names[i]}', fontsize=13, fontweight='bold')
            axes[i].legend(loc='upper left')
            axes[i].grid(alpha=0.3)
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Calibration curves saved to {save_path}")
    
    def threshold_analysis(self, save_path='reports/figures/threshold_analysis.png'):
        """Analyze different classification thresholds."""
        fig, axes = plt.subplots(1, self.num_classes, figsize=(6*self.num_classes, 5))
        if self.num_classes == 1:
            axes = [axes]
        
        for i in range(self.num_classes):
            binary_labels = (self.all_labels == i).astype(int)
            binary_probs = self.all_probabilities[:, i]
            
            fpr, tpr, thresholds = roc_curve(binary_labels, binary_probs)
            
            # Calculate Youden's J statistic
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_idx]
            
            # Plot
            axes[i].plot(thresholds, tpr, label='True Positive Rate', linewidth=2)
            axes[i].plot(thresholds, fpr, label='False Positive Rate', linewidth=2)
            axes[i].plot(thresholds, tpr - fpr, label="Youden's J", linewidth=2, linestyle='--')
            axes[i].axvline(best_threshold, color='red', linestyle=':', linewidth=2,
                          label=f'Optimal ({best_threshold:.3f})')
            axes[i].set_xlabel('Threshold', fontsize=11)
            axes[i].set_ylabel('Rate', fontsize=11)
            axes[i].set_title(f'{self.class_names[i]}', fontsize=13, fontweight='bold')
            axes[i].legend(loc='best')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Threshold analysis saved to {save_path}")
    
    def generate_report(self, metrics, save_path='reports/metrics/evaluation_report.json'):
        """Save comprehensive evaluation report."""
        report = {
            'overall_metrics': {
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_micro': metrics['f1_micro'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'auc_roc_macro': metrics['auc_roc_macro'],
            },
            'per_class_metrics': {}
        }
        
        for class_name in self.class_names:
            report['per_class_metrics'][class_name] = {
                'precision': metrics['precision_per_class'][class_name],
                'recall': metrics['recall_per_class'][class_name],
                'f1': metrics['f1_per_class'][class_name],
                'sensitivity': metrics['sensitivity_per_class'][class_name],
                'specificity': metrics['specificity_per_class'][class_name],
                'auc_roc': metrics['auc_roc_per_class'][class_name],
            }
        
        # Classification report
        report['classification_report'] = classification_report(
            self.all_labels, self.all_predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved to {save_path}")
    
    def evaluate(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*70)
        print(" " * 20 + "COMPREHENSIVE EVALUATION")
        print("="*70)
        
        # Get predictions
        self.get_predictions()
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Generate visualizations
        ensure_dir('reports/figures')
        ensure_dir('reports/metrics')
        
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_calibration()
        self.threshold_analysis()
        
        # Generate report
        self.generate_report(metrics)
        
        print("\n" + "="*70)
        print(" " * 20 + "EVALUATION COMPLETE")
        print("="*70)
        print("\nAll results saved to reports/")
        
        return metrics


def main():
    """Main evaluation function."""
    config = load_config()
    set_seed(config['seed'])
    device = get_device(config)
    
    print(f"\nUsing device: {device}")
    
    # Load test data
    print("\n📊 Loading test data...")
    _, _, test_loader, class_names = create_dataloaders(
        config,
        train_csv="data/processed/train_split.csv",
        val_csv="data/processed/val_split.csv",
        test_csv="data/processed/test_split.csv"
    )
    
    # Load best model
    print("\n🏗️  Loading best model...")
    model = create_model(config)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, class_names, device)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    print(f"\n✅ Comprehensive evaluation complete!")


if __name__ == "__main__":
    main()
