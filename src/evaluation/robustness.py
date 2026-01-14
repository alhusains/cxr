"""
Robustness and Distribution Shift Evaluation (Task 3).

Evaluates model performance under:
- Clinical corruptions (scanner variance, resolution, noise)
- Distribution shift analysis
- Embedding drift
- Performance degradation analysis
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import cosine
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import load_config, get_device, ensure_dir, set_seed
from src.data.dataset import ChestXRayDataset
from src.models.model import create_model


class RobustnessEvaluator:
    """
    Evaluate model robustness under distribution shift.
    
    Args:
        model: Trained model
        test_csv: Path to test split CSV
        class_names: List of class names
        device: Device to run on
        config: Configuration dict
    """
    
    def __init__(self, model, test_csv, class_names, device, config):
        self.model = model
        self.test_csv = test_csv
        self.class_names = class_names
        self.device = device
        self.config = config
        self.num_classes = len(class_names)
        
        # Create feature extractor (model without classifier)
        self.feature_extractor = self._create_feature_extractor()
    
    def _create_feature_extractor(self):
        """Create feature extractor from model backbone."""
        # Clone model and remove classifier to get embeddings
        feature_model = nn.Sequential(*list(self.model.backbone.children()))
        return feature_model.eval()
    
    def get_corruption_transforms(self, corruption_type='clinical'):
        """
        Get transforms for different corruption types.
        
        Args:
            corruption_type: Type of corruption to apply
            
        Returns:
            Albumentations transform pipeline
        """
        target_size = self.config['preprocessing']['target_size']
        mean = self.config['preprocessing']['normalize_mean']
        std = self.config['preprocessing']['normalize_std']
        
        if corruption_type == 'clinical':
            # Clinical corruptions: scanner variance, image quality
            transforms = A.Compose([
                # Scanner variance
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                # Low-resolution portable scanners
                A.GaussianBlur(blur_limit=(3, 7), p=0.7),
                # Sensor noise
                A.GaussNoise(var_limit=(0.01, 0.03), mean=0, p=0.7),
                # Resolution degradation
                A.Downscale(scale_min=0.5, scale_max=0.7, p=0.5),
                # Standard preprocessing
                A.LongestMaxSize(max_size=target_size[0]),
                A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        
        elif corruption_type == 'severe':
            # Severe corruptions: extreme conditions
            transforms = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=1.0
                ),
                A.GaussianBlur(blur_limit=(5, 11), p=1.0),
                A.GaussNoise(var_limit=(0.03, 0.05), mean=0, p=1.0),
                A.LongestMaxSize(max_size=target_size[0]),
                A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        
        elif corruption_type == 'clean':
            # Clean/original test set
            transforms = A.Compose([
                A.LongestMaxSize(max_size=target_size[0]),
                A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1],
                            border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        
        return transforms
    
    def get_predictions_and_embeddings(self, dataloader, extract_embeddings=True):
        """
        Get predictions and embeddings from model.
        
        Args:
            dataloader: DataLoader to evaluate
            extract_embeddings: Whether to extract embeddings
            
        Returns:
            labels, predictions, probabilities, embeddings (if requested)
        """
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Get embeddings if requested
                if extract_embeddings:
                    embeddings = self.feature_extractor(images)
                    # Global average pooling if needed
                    if len(embeddings.shape) > 2:
                        embeddings = F.adaptive_avg_pool2d(embeddings, (1, 1))
                        embeddings = embeddings.view(embeddings.size(0), -1)
                    all_embeddings.extend(embeddings.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        if extract_embeddings:
            all_embeddings = np.array(all_embeddings)
            return all_labels, all_predictions, all_probabilities, all_embeddings
        else:
            return all_labels, all_predictions, all_probabilities, None
    
    def compute_metrics(self, labels, predictions, probabilities):
        """Compute evaluation metrics."""
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = (predictions == labels).mean()
        
        # Per-class F1 and AUC-ROC
        f1_scores = []
        auc_scores = []
        
        for i in range(self.num_classes):
            binary_labels = (labels == i).astype(int)
            binary_preds = (predictions == i).astype(int)
            binary_probs = probabilities[:, i]
            
            f1 = f1_score(binary_labels, binary_preds, zero_division=0)
            f1_scores.append(f1)
            
            try:
                auc = roc_auc_score(binary_labels, binary_probs)
                auc_scores.append(auc)
            except:
                auc_scores.append(0.0)
        
        metrics['f1_per_class'] = {self.class_names[i]: f1_scores[i] for i in range(self.num_classes)}
        metrics['auc_per_class'] = {self.class_names[i]: auc_scores[i] for i in range(self.num_classes)}
        metrics['f1_macro'] = np.mean(f1_scores)
        metrics['auc_macro'] = np.mean(auc_scores)
        
        return metrics
    
    def compute_embedding_drift(self, embeddings_clean, embeddings_corrupted):
        """
        Compute embedding drift using multiple metrics.
        
        Args:
            embeddings_clean: Embeddings from clean test set
            embeddings_corrupted: Embeddings from corrupted test set
            
        Returns:
            Dictionary of drift metrics
        """
        drift_metrics = {}
        
        # 1. Cosine distance between centroids
        centroid_clean = np.mean(embeddings_clean, axis=0)
        centroid_corrupted = np.mean(embeddings_corrupted, axis=0)
        
        cosine_dist = cosine(centroid_clean, centroid_corrupted)
        drift_metrics['cosine_distance'] = float(cosine_dist)
        
        # 2. Mean pairwise distance shift
        mean_norm_clean = np.mean(np.linalg.norm(embeddings_clean, axis=1))
        mean_norm_corrupted = np.mean(np.linalg.norm(embeddings_corrupted, axis=1))
        drift_metrics['norm_shift'] = float(abs(mean_norm_clean - mean_norm_corrupted))
        
        # 3. Fréchet Distance (simplified)
        mu_clean = np.mean(embeddings_clean, axis=0)
        mu_corrupted = np.mean(embeddings_corrupted, axis=0)
        
        sigma_clean = np.cov(embeddings_clean, rowvar=False)
        sigma_corrupted = np.cov(embeddings_corrupted, rowvar=False)
        
        # Fréchet distance
        diff = mu_clean - mu_corrupted
        covmean = linalg.sqrtm(sigma_clean.dot(sigma_corrupted))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum(diff**2) + np.trace(sigma_clean + sigma_corrupted - 2*covmean)
        drift_metrics['frechet_distance'] = float(fid)
        
        return drift_metrics
    
    def analyze_confidence_shift(self, probs_clean, probs_corrupted, 
                                 save_path='reports/figures/confidence_shift.png'):
        """Analyze and visualize confidence shifts."""
        # Get max probabilities (confidence)
        conf_clean = np.max(probs_clean, axis=1)
        conf_corrupted = np.max(probs_corrupted, axis=1)
        
        # Statistics
        conf_metrics = {
            'mean_confidence_clean': float(np.mean(conf_clean)),
            'mean_confidence_corrupted': float(np.mean(conf_corrupted)),
            'std_confidence_clean': float(np.std(conf_clean)),
            'std_confidence_corrupted': float(np.std(conf_corrupted)),
            'confidence_drop': float(np.mean(conf_clean) - np.mean(conf_corrupted))
        }
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram comparison
        axes[0].hist(conf_clean, bins=50, alpha=0.6, label='Clean Test Set', 
                    color='blue', edgecolor='black')
        axes[0].hist(conf_corrupted, bins=50, alpha=0.6, label='Corrupted Test Set',
                    color='red', edgecolor='black')
        axes[0].set_xlabel('Prediction Confidence (Max Probability)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Confidence Distribution Shift', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot([conf_clean, conf_corrupted], 
                       labels=['Clean', 'Corrupted'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
        axes[1].set_ylabel('Prediction Confidence', fontsize=12)
        axes[1].set_title('Confidence Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confidence shift plot saved to {save_path}")
        
        return conf_metrics
    
    def evaluate_robustness(self):
        """Run complete robustness evaluation."""
        print("\n" + "="*70)
        print(" " * 15 + "ROBUSTNESS EVALUATION (TASK 3)")
        print("="*70)
        
        results = {}
        
        # 1. Clean test set
        print("\n📊 Evaluating on CLEAN test set...")
        clean_transform = self.get_corruption_transforms('clean')
        clean_dataset = ChestXRayDataset(self.test_csv, transform=clean_transform)
        clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        labels_clean, preds_clean, probs_clean, emb_clean = \
            self.get_predictions_and_embeddings(clean_loader)
        
        metrics_clean = self.compute_metrics(labels_clean, preds_clean, probs_clean)
        results['clean'] = metrics_clean
        
        print(f"  Accuracy: {metrics_clean['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics_clean['f1_macro']:.4f}")
        print(f"  AUC-ROC (macro): {metrics_clean['auc_macro']:.4f}")
        
        # 2. Clinical corruptions
        print("\n📊 Evaluating on CLINICAL CORRUPTIONS (scanner variance, blur, noise)...")
        clinical_transform = self.get_corruption_transforms('clinical')
        clinical_dataset = ChestXRayDataset(self.test_csv, transform=clinical_transform)
        clinical_loader = DataLoader(clinical_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        labels_clinical, preds_clinical, probs_clinical, emb_clinical = \
            self.get_predictions_and_embeddings(clinical_loader)
        
        metrics_clinical = self.compute_metrics(labels_clinical, preds_clinical, probs_clinical)
        results['clinical_corruptions'] = metrics_clinical
        
        print(f"  Accuracy: {metrics_clinical['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics_clinical['f1_macro']:.4f}")
        print(f"  AUC-ROC (macro): {metrics_clinical['auc_macro']:.4f}")
        
        # 3. Severe corruptions
        print("\n📊 Evaluating on SEVERE CORRUPTIONS (extreme conditions)...")
        severe_transform = self.get_corruption_transforms('severe')
        severe_dataset = ChestXRayDataset(self.test_csv, transform=severe_transform)
        severe_loader = DataLoader(severe_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        labels_severe, preds_severe, probs_severe, emb_severe = \
            self.get_predictions_and_embeddings(severe_loader)
        
        metrics_severe = self.compute_metrics(labels_severe, preds_severe, probs_severe)
        results['severe_corruptions'] = metrics_severe
        
        print(f"  Accuracy: {metrics_severe['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics_severe['f1_macro']:.4f}")
        print(f"  AUC-ROC (macro): {metrics_severe['auc_macro']:.4f}")
        
        # 4. Performance drift analysis
        print("\n" + "="*70)
        print("PERFORMANCE DRIFT ANALYSIS")
        print("="*70)
        
        drift_clinical = {
            'accuracy_drop': metrics_clean['accuracy'] - metrics_clinical['accuracy'],
            'f1_drop': metrics_clean['f1_macro'] - metrics_clinical['f1_macro'],
            'auc_drop': metrics_clean['auc_macro'] - metrics_clinical['auc_macro'],
        }
        
        drift_severe = {
            'accuracy_drop': metrics_clean['accuracy'] - metrics_severe['accuracy'],
            'f1_drop': metrics_clean['f1_macro'] - metrics_severe['f1_macro'],
            'auc_drop': metrics_clean['auc_macro'] - metrics_severe['auc_macro'],
        }
        
        print(f"\nClinical Corruptions:")
        print(f"  Accuracy drop: {drift_clinical['accuracy_drop']:.4f} ({drift_clinical['accuracy_drop']/metrics_clean['accuracy']*100:.1f}%)")
        print(f"  F1 drop: {drift_clinical['f1_drop']:.4f} ({drift_clinical['f1_drop']/metrics_clean['f1_macro']*100:.1f}%)")
        print(f"  AUC-ROC drop: {drift_clinical['auc_drop']:.4f} ({drift_clinical['auc_drop']/metrics_clean['auc_macro']*100:.1f}%)")
        
        print(f"\nSevere Corruptions:")
        print(f"  Accuracy drop: {drift_severe['accuracy_drop']:.4f} ({drift_severe['accuracy_drop']/metrics_clean['accuracy']*100:.1f}%)")
        print(f"  F1 drop: {drift_severe['f1_drop']:.4f} ({drift_severe['f1_drop']/metrics_clean['f1_macro']*100:.1f}%)")
        print(f"  AUC-ROC drop: {drift_severe['auc_drop']:.4f} ({drift_severe['auc_drop']/metrics_clean['auc_macro']*100:.1f}%)")
        
        results['performance_drift_clinical'] = drift_clinical
        results['performance_drift_severe'] = drift_severe
        
        # 5. Embedding drift
        print("\n" + "="*70)
        print("EMBEDDING DRIFT ANALYSIS")
        print("="*70)
        
        embedding_drift_clinical = self.compute_embedding_drift(emb_clean, emb_clinical)
        embedding_drift_severe = self.compute_embedding_drift(emb_clean, emb_severe)
        
        print(f"\nClinical Corruptions:")
        print(f"  Cosine distance: {embedding_drift_clinical['cosine_distance']:.4f}")
        print(f"  Norm shift: {embedding_drift_clinical['norm_shift']:.4f}")
        print(f"  Fréchet distance: {embedding_drift_clinical['frechet_distance']:.2f}")
        
        print(f"\nSevere Corruptions:")
        print(f"  Cosine distance: {embedding_drift_severe['cosine_distance']:.4f}")
        print(f"  Norm shift: {embedding_drift_severe['norm_shift']:.4f}")
        print(f"  Fréchet distance: {embedding_drift_severe['frechet_distance']:.2f}")
        
        results['embedding_drift_clinical'] = embedding_drift_clinical
        results['embedding_drift_severe'] = embedding_drift_severe
        
        # 6. Confidence shift analysis
        print("\n" + "="*70)
        print("CONFIDENCE SHIFT ANALYSIS")
        print("="*70)
        
        ensure_dir('reports/figures')
        conf_metrics_clinical = self.analyze_confidence_shift(
            probs_clean, probs_clinical,
            'reports/figures/confidence_shift_clinical.png'
        )
        conf_metrics_severe = self.analyze_confidence_shift(
            probs_clean, probs_severe,
            'reports/figures/confidence_shift_severe.png'
        )
        
        print(f"\nClinical Corruptions:")
        print(f"  Mean confidence (clean): {conf_metrics_clinical['mean_confidence_clean']:.4f}")
        print(f"  Mean confidence (corrupted): {conf_metrics_clinical['mean_confidence_corrupted']:.4f}")
        print(f"  Confidence drop: {conf_metrics_clinical['confidence_drop']:.4f}")
        
        print(f"\nSevere Corruptions:")
        print(f"  Mean confidence (clean): {conf_metrics_severe['mean_confidence_clean']:.4f}")
        print(f"  Mean confidence (corrupted): {conf_metrics_severe['mean_confidence_corrupted']:.4f}")
        print(f"  Confidence drop: {conf_metrics_severe['confidence_drop']:.4f}")
        
        results['confidence_shift_clinical'] = conf_metrics_clinical
        results['confidence_shift_severe'] = conf_metrics_severe
        
        # 7. Mitigation recommendations
        print("\n" + "="*70)
        print("MITIGATION RECOMMENDATIONS")
        print("="*70)
        
        recommendations = self.generate_recommendations(results)
        for rec in recommendations:
            print(f"  • {rec}")
        
        results['recommendations'] = recommendations
        
        # Save results
        ensure_dir('reports/metrics')
        with open('reports/metrics/robustness_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print(" " * 15 + "ROBUSTNESS EVALUATION COMPLETE")
        print("="*70)
        print("\n✓ Results saved to reports/metrics/robustness_evaluation.json")
        print("✓ Visualizations saved to reports/figures/confidence_shift_*.png")
        
        return results
    
    def generate_recommendations(self, results):
        """Generate mitigation recommendations based on results."""
        recommendations = []
        
        # Check performance drops
        f1_drop_clinical = results['performance_drift_clinical']['f1_drop']
        f1_drop_severe = results['performance_drift_severe']['f1_drop']
        
        if f1_drop_clinical > 0.05:
            recommendations.append(
                "Significant performance drop on clinical corruptions detected. "
                "Recommendation: Augment training data with brightness/contrast variations."
            )
        
        if f1_drop_severe > 0.15:
            recommendations.append(
                "Severe performance degradation under extreme conditions. "
                "Recommendation: Implement test-time augmentation (TTA) for robust predictions."
            )
        
        # Check embedding drift
        cosine_dist = results['embedding_drift_clinical']['cosine_distance']
        if cosine_dist > 0.1:
            recommendations.append(
                "High embedding drift detected. "
                "Recommendation: Consider domain adaptation techniques or self-supervised pretraining."
            )
        
        # Check confidence calibration
        conf_drop = results['confidence_shift_clinical']['confidence_drop']
        if abs(conf_drop) > 0.1:
            recommendations.append(
                "Model confidence shifts significantly under distribution shift. "
                "Recommendation: Apply temperature scaling or ensemble methods for better calibration."
            )
        
        # General recommendations
        recommendations.extend([
            "Collect diverse training data from multiple scanners/hospitals for better generalization.",
            "Implement uncertainty quantification to flag low-confidence predictions for human review.",
            "Regular recalibration with new data from deployment environment."
        ])
        
        return recommendations


def main():
    """Main robustness evaluation function."""
    config = load_config()
    set_seed(config['seed'])
    device = get_device(config)
    
    print(f"\nUsing device: {device}")
    
    # Load class names
    import pandas as pd
    test_df = pd.read_csv('data/processed/test_split.csv')
    class_names = sorted(test_df['class'].unique())
    
    # Load best model
    print("\n🏗️  Loading best model...")
    model = create_model(config)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
    
    # Create evaluator
    evaluator = RobustnessEvaluator(
        model=model,
        test_csv='data/processed/test_split.csv',
        class_names=class_names,
        device=device,
        config=config
    )
    
    # Run evaluation
    results = evaluator.evaluate_robustness()
    
    print("\n✅ Robustness evaluation complete!")


if __name__ == "__main__":
    main()
