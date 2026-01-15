"""
Compute and save training distribution statistics for data drift detection.

This script analyzes the training dataset to extract statistical properties
that will be used as reference for drift detection during inference.

Run this script after training to generate the reference statistics file.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import load_config


def compute_training_statistics(data_dir: Path, sample_size: int = 1000, seed: int = 42):
    """
    Compute statistical properties of training images.
    
    Args:
        data_dir: Path to training data directory
        sample_size: Number of images to sample for statistics
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with training distribution statistics
    """
    np.random.seed(seed)
    
    print("Computing training distribution statistics...")
    print(f"Data directory: {data_dir}")
    
    # Collect all training image paths
    train_images = []
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            train_images.extend(list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
    
    print(f"Found {len(train_images)} training images")
    
    # Sample images if dataset is large
    if len(train_images) > sample_size:
        print(f"Sampling {sample_size} images for efficiency")
        sampled_images = np.random.choice(train_images, size=sample_size, replace=False)
    else:
        sampled_images = train_images
    
    # Collect statistics
    mean_intensities = []
    std_intensities = []
    aspect_ratios = []
    
    print("Analyzing images...")
    for img_path in tqdm(sampled_images):
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to grayscale for intensity analysis
            img_gray = np.array(img.convert('L'))
            
            # Compute statistics
            mean_intensities.append(float(np.mean(img_gray)))
            std_intensities.append(float(np.std(img_gray)))
            aspect_ratios.append(img.size[0] / img.size[1])
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Aggregate statistics
    mean_intensities = np.array(mean_intensities)
    std_intensities = np.array(std_intensities)
    aspect_ratios = np.array(aspect_ratios)
    
    stats = {
        # Mean intensity statistics
        'mean_intensity': float(np.mean(mean_intensities)),
        'std_intensity': float(np.std(mean_intensities)),
        'median_intensity': float(np.median(mean_intensities)),
        'q25_intensity': float(np.percentile(mean_intensities, 25)),
        'q75_intensity': float(np.percentile(mean_intensities, 75)),
        
        # Standard deviation statistics
        'mean_std': float(np.mean(std_intensities)),
        'std_of_stds': float(np.std(std_intensities)),
        
        # Variability measure
        'std_of_means': float(np.std(mean_intensities)),
        
        # Aspect ratio statistics
        'mean_aspect_ratio': float(np.mean(aspect_ratios)),
        'std_aspect_ratio': float(np.std(aspect_ratios)),
        
        # Sample of intensity values for KS test
        'intensity_samples': mean_intensities.tolist()[:500],  # Keep 500 samples
        
        # Metadata
        'sample_size': len(mean_intensities),
        'total_images': len(train_images),
        'seed': seed,
        
        # Placeholder for confidence (to be filled from training logs if available)
        'mean_confidence': 0.85  # Default estimate
    }
    
    # Print summary
    print("\n" + "="*60)
    print("Training Distribution Statistics Summary")
    print("="*60)
    print(f"Sample size: {stats['sample_size']:,} images")
    print(f"Total training images: {stats['total_images']:,}")
    print(f"\nIntensity Statistics (0-255 scale):")
    print(f"  Mean: {stats['mean_intensity']:.2f} ± {stats['std_intensity']:.2f}")
    print(f"  Median: {stats['median_intensity']:.2f}")
    print(f"  Q25-Q75: [{stats['q25_intensity']:.2f}, {stats['q75_intensity']:.2f}]")
    print(f"\nContrast Statistics:")
    print(f"  Mean std: {stats['mean_std']:.2f} ± {stats['std_of_stds']:.2f}")
    print(f"\nAspect Ratio:")
    print(f"  Mean: {stats['mean_aspect_ratio']:.3f} ± {stats['std_aspect_ratio']:.3f}")
    print("="*60)
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute training distribution statistics for drift detection"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/train',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/training_distribution_stats.npy',
        help='Output path for statistics file'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=1000,
        help='Number of images to sample for statistics'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Compute statistics
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure the training data is available.")
        return
    
    stats = compute_training_statistics(
        data_dir=data_dir,
        sample_size=args.sample_size,
        seed=args.seed
    )
    
    # Save statistics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, stats)
    
    print(f"\n✓ Statistics saved to: {output_path}")
    print("\nThese statistics will be used for drift detection in the API.")
    print("The API will compare incoming data distributions against these reference values.")


if __name__ == "__main__":
    main()
