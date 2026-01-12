"""
Enhanced EDA: Edge/Texture Analysis and Artifact Detection

Analyzes:
- Edge characteristics (Canny edge detection)
- Texture features (Local Binary Patterns)
- Exposure quality (under/over-exposed detection)
- Common imaging artifacts
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
from skimage.feature import local_binary_pattern

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import ensure_dir


def analyze_edge_characteristics(data_path="data/raw", output_path="reports/figures", n_samples=100):
    """Analyze edge characteristics using Canny edge detection."""
    print("\n" + "="*60)
    print("Edge Characteristics Analysis")
    print("="*60)
    
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Find dataset root
    dataset_root = data_path
    if not (dataset_root / "train").exists():
        possible_roots = [data_path / "chest_xray", data_path / "ChestXRay2017"]
        for root in possible_roots:
            if root.exists() and (root / "train").exists():
                dataset_root = root
                break
    
    edge_stats = []
    
    for split in ['train']:
        split_path = dataset_root / split
        if not split_path.exists():
            continue
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob("*.jpg"))[:n_samples]
            
            print(f"Analyzing {class_name}...")
            
            for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize for consistent analysis
                    img = cv2.resize(img, (512, 512))
                    
                    # Canny edge detection
                    edges = cv2.Canny(img, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    
                    # Gradient magnitude
                    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                    avg_gradient = np.mean(gradient_mag)
                    
                    edge_stats.append({
                        'class': class_name,
                        'edge_density': edge_density,
                        'avg_gradient': avg_gradient,
                    })
                    
                except Exception as e:
                    continue
    
    # Visualize
    df = pd.DataFrame(edge_stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Edge density
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        axes[0].hist(class_df['edge_density'], bins=30, alpha=0.6, label=class_name, edgecolor='black')
    axes[0].set_title('Edge Density Distribution', fontweight='bold')
    axes[0].set_xlabel('Edge Density (proportion of edge pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Gradient magnitude
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        axes[1].hist(class_df['avg_gradient'], bins=30, alpha=0.6, label=class_name, edgecolor='black')
    axes[1].set_title('Average Gradient Magnitude', fontweight='bold')
    axes[1].set_xlabel('Average Gradient')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'edge_characteristics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'edge_characteristics.png'}")
    plt.close()
    
    # Stats
    print("\nEdge statistics by class:")
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        print(f"  {class_name}:")
        print(f"    Edge density: {class_df['edge_density'].mean():.4f} ± {class_df['edge_density'].std():.4f}")
        print(f"    Avg gradient: {class_df['avg_gradient'].mean():.2f} ± {class_df['avg_gradient'].std():.2f}")
    
    return df


def analyze_texture_features(data_path="data/raw", output_path="reports/figures", n_samples=100):
    """Analyze texture using Local Binary Patterns."""
    print("\n" + "="*60)
    print("Texture Characteristics Analysis (LBP)")
    print("="*60)
    
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Find dataset root
    dataset_root = data_path
    if not (dataset_root / "train").exists():
        possible_roots = [data_path / "chest_xray", data_path / "ChestXRay2017"]
        for root in possible_roots:
            if root.exists() and (root / "train").exists():
                dataset_root = root
                break
    
    texture_stats = []
    
    for split in ['train']:
        split_path = dataset_root / split
        if not split_path.exists():
            continue
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob("*.jpg"))[:n_samples]
            
            print(f"Analyzing {class_name}...")
            
            for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize for consistent analysis
                    img = cv2.resize(img, (256, 256))
                    
                    # LBP
                    lbp = local_binary_pattern(img, 8, 1, method='uniform')
                    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
                    lbp_entropy = -np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0]))
                    
                    # GLCM-like features (contrast)
                    contrast = np.std(img)
                    
                    texture_stats.append({
                        'class': class_name,
                        'lbp_entropy': lbp_entropy,
                        'contrast': contrast,
                    })
                    
                except Exception as e:
                    continue
    
    # Visualize
    df = pd.DataFrame(texture_stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # LBP entropy
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        axes[0].hist(class_df['lbp_entropy'], bins=30, alpha=0.6, label=class_name, edgecolor='black')
    axes[0].set_title('LBP Entropy (Texture Complexity)', fontweight='bold')
    axes[0].set_xlabel('LBP Entropy')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Contrast
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        axes[1].hist(class_df['contrast'], bins=30, alpha=0.6, label=class_name, edgecolor='black')
    axes[1].set_title('Image Contrast', fontweight='bold')
    axes[1].set_xlabel('Contrast (Std Dev)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'texture_characteristics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'texture_characteristics.png'}")
    plt.close()
    
    # Stats
    print("\nTexture statistics by class:")
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        print(f"  {class_name}:")
        print(f"    LBP entropy: {class_df['lbp_entropy'].mean():.4f} ± {class_df['lbp_entropy'].std():.4f}")
        print(f"    Contrast: {class_df['contrast'].mean():.2f} ± {class_df['contrast'].std():.2f}")
    
    return df


def detect_exposure_artifacts(data_path="data/raw", output_path="reports/figures", n_samples=200):
    """Detect under/over-exposed images and other artifacts."""
    print("\n" + "="*60)
    print("Exposure & Artifact Detection")
    print("="*60)
    
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Find dataset root
    dataset_root = data_path
    if not (dataset_root / "train").exists():
        possible_roots = [data_path / "chest_xray", data_path / "ChestXRay2017"]
        for root in possible_roots:
            if root.exists() and (root / "train").exists():
                dataset_root = root
                break
    
    exposure_stats = []
    underexposed_samples = []
    overexposed_samples = []
    
    for split in ['train', 'test']:
        split_path = dataset_root / split
        if not split_path.exists():
            continue
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob("*.jpg"))[:n_samples]
            
            print(f"Analyzing {class_name} ({split})...")
            
            for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    mean_intensity = np.mean(img)
                    std_intensity = np.std(img)
                    
                    # Exposure detection thresholds
                    is_underexposed = mean_intensity < 70
                    is_overexposed = mean_intensity > 200
                    
                    # Dynamic range
                    dynamic_range = np.max(img) - np.min(img)
                    low_dynamic_range = dynamic_range < 100
                    
                    exposure_stats.append({
                        'class': class_name,
                        'split': split,
                        'mean_intensity': mean_intensity,
                        'std_intensity': std_intensity,
                        'dynamic_range': dynamic_range,
                        'underexposed': is_underexposed,
                        'overexposed': is_overexposed,
                        'low_dynamic_range': low_dynamic_range,
                    })
                    
                    # Collect samples for visualization
                    if is_underexposed and len(underexposed_samples) < 4:
                        underexposed_samples.append((img_path, mean_intensity))
                    if is_overexposed and len(overexposed_samples) < 4:
                        overexposed_samples.append((img_path, mean_intensity))
                    
                except Exception as e:
                    continue
    
    df = pd.DataFrame(exposure_stats)
    
    # Statistics
    print(f"\nExposure artifacts detected:")
    print(f"  Underexposed images: {df['underexposed'].sum()} ({df['underexposed'].mean()*100:.1f}%)")
    print(f"  Overexposed images: {df['overexposed'].sum()} ({df['overexposed'].mean()*100:.1f}%)")
    print(f"  Low dynamic range: {df['low_dynamic_range'].sum()} ({df['low_dynamic_range'].mean()*100:.1f}%)")
    
    print("\nPer-class breakdown:")
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        print(f"  {class_name}:")
        print(f"    Underexposed: {class_df['underexposed'].sum()} ({class_df['underexposed'].mean()*100:.1f}%)")
        print(f"    Overexposed: {class_df['overexposed'].sum()} ({class_df['overexposed'].mean()*100:.1f}%)")
    
    # Visualize exposure artifacts
    if underexposed_samples or overexposed_samples:
        n_under = len(underexposed_samples)
        n_over = len(overexposed_samples)
        total_samples = max(n_under, n_over)
        
        if total_samples > 0:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Underexposed
            for idx in range(4):
                if idx < n_under:
                    img = cv2.imread(str(underexposed_samples[idx][0]), cv2.IMREAD_GRAYSCALE)
                    axes[0, idx].imshow(img, cmap='gray')
                    axes[0, idx].set_title(f'Mean: {underexposed_samples[idx][1]:.1f}', fontsize=10)
                axes[0, idx].axis('off')
            axes[0, 0].text(-0.3, 0.5, 'Underexposed', transform=axes[0, 0].transAxes,
                          fontsize=14, fontweight='bold', rotation=90, va='center')
            
            # Overexposed
            for idx in range(4):
                if idx < n_over:
                    img = cv2.imread(str(overexposed_samples[idx][0]), cv2.IMREAD_GRAYSCALE)
                    axes[1, idx].imshow(img, cmap='gray')
                    axes[1, idx].set_title(f'Mean: {overexposed_samples[idx][1]:.1f}', fontsize=10)
                axes[1, idx].axis('off')
            axes[1, 0].text(-0.3, 0.5, 'Overexposed', transform=axes[1, 0].transAxes,
                          fontsize=14, fontweight='bold', rotation=90, va='center')
            
            plt.suptitle('Exposure Artifacts Examples', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'exposure_artifacts.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path / 'exposure_artifacts.png'}")
            plt.close()
    
    # Histogram visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Mean intensity distribution
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        axes[0].hist(class_df['mean_intensity'], bins=50, alpha=0.6, label=class_name, edgecolor='black')
    axes[0].axvline(70, color='red', linestyle='--', linewidth=2, label='Underexposed threshold')
    axes[0].axvline(200, color='orange', linestyle='--', linewidth=2, label='Overexposed threshold')
    axes[0].set_title('Mean Intensity Distribution', fontweight='bold')
    axes[0].set_xlabel('Mean Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Dynamic range
    for class_name in df['class'].unique():
        class_df = df[df['class'] == class_name]
        axes[1].hist(class_df['dynamic_range'], bins=50, alpha=0.6, label=class_name, edgecolor='black')
    axes[1].axvline(100, color='red', linestyle='--', linewidth=2, label='Low dynamic range threshold')
    axes[1].set_title('Dynamic Range Distribution', fontweight='bold')
    axes[1].set_xlabel('Dynamic Range (max - min)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'exposure_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'exposure_analysis.png'}")
    plt.close()
    
    return df


def main():
    """Run enhanced EDA analyses."""
    print("\n" + "="*70)
    print(" " * 15 + "ENHANCED EDA: EDGES, TEXTURE & ARTIFACTS")
    print("="*70)
    
    ensure_dir("reports/figures")
    ensure_dir("reports/metrics")
    
    # Run analyses
    edge_df = analyze_edge_characteristics()
    texture_df = analyze_texture_features()
    exposure_df = detect_exposure_artifacts()
    
    # Save combined results
    edge_df.to_csv('reports/metrics/edge_analysis.csv', index=False)
    texture_df.to_csv('reports/metrics/texture_analysis.csv', index=False)
    exposure_df.to_csv('reports/metrics/exposure_analysis.csv', index=False)
    
    print("\n" + "="*70)
    print(" " * 20 + "ENHANCED EDA COMPLETE!")
    print("="*70)
    print("\nNew results saved to:")
    print("  - reports/figures/edge_characteristics.png")
    print("  - reports/figures/texture_characteristics.png")
    print("  - reports/figures/exposure_artifacts.png")
    print("  - reports/figures/exposure_analysis.png")
    print("  - reports/metrics/*_analysis.csv")


if __name__ == "__main__":
    main()
