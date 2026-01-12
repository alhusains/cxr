"""
Exploratory Data Analysis for Chest X-Ray Dataset

Analyzes:
- Class distribution across splits
- Image dimensions and aspect ratios
- Pixel intensity statistics
- Data quality issues (duplicates, corrupt files)
- Common artifacts
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import ensure_dir


class ChestXRayEDA:
    """Comprehensive EDA for chest X-ray dataset."""
    
    def __init__(self, data_path="data/raw", output_path="reports/figures"):
        """
        Initialize EDA analyzer.
        
        Args:
            data_path: Path to raw data directory
            output_path: Path to save visualizations
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        ensure_dir(self.output_path)
        
        self.image_info = []
        self.corrupt_files = []
        
        # Find dataset root
        self.dataset_root = self._find_dataset_root()
        if not self.dataset_root:
            raise ValueError(f"Could not find dataset in {data_path}")
    
    def _find_dataset_root(self):
        """Find the actual dataset root directory."""
        possible_roots = [
            self.data_path,
            self.data_path / "chest_xray",
            self.data_path / "ChestXRay2017",
        ]
        
        for root in possible_roots:
            if root.exists():
                subdirs = [d.name.lower() for d in root.iterdir() if d.is_dir()]
                if any(split in subdirs for split in ['train', 'test', 'val']):
                    return root
        return None
    
    def scan_dataset(self):
        """Scan entire dataset and collect metadata."""
        print("\n" + "="*60)
        print("Scanning Dataset")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_root / split
            
            # Handle case variations
            if not split_path.exists():
                split_path = self.dataset_root / split.upper()
            if not split_path.exists():
                print(f"⚠ Split '{split}' not found, skipping")
                continue
            
            print(f"\nScanning {split}/ ...")
            
            # Get class directories
            class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.jpeg")) + \
                             list(class_dir.glob("*.jpg")) + \
                             list(class_dir.glob("*.png"))
                
                print(f"  {class_name}: {len(image_files)} images")
                
                for img_path in tqdm(image_files, desc=f"  Processing {class_name}", leave=False):
                    self._process_image(img_path, split, class_name)
        
        print(f"\n✓ Scanned {len(self.image_info)} images")
        if self.corrupt_files:
            print(f"⚠ Found {len(self.corrupt_files)} corrupt files")
    
    def _process_image(self, img_path, split, class_name):
        """Process a single image and collect metadata."""
        try:
            # Read image
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Collect metadata
            info = {
                'path': str(img_path),
                'split': split,
                'class': class_name,
                'width': img.width,
                'height': img.height,
                'aspect_ratio': img.width / img.height,
                'channels': len(img_array.shape),
                'mode': img.mode,
                'file_size': img_path.stat().st_size,
            }
            
            # Pixel intensity statistics
            if len(img_array.shape) == 2:  # Grayscale
                info['mean_intensity'] = np.mean(img_array)
                info['std_intensity'] = np.std(img_array)
                info['min_intensity'] = np.min(img_array)
                info['max_intensity'] = np.max(img_array)
            else:  # RGB or RGBA
                info['mean_intensity'] = np.mean(img_array)
                info['std_intensity'] = np.std(img_array)
                info['min_intensity'] = np.min(img_array)
                info['max_intensity'] = np.max(img_array)
            
            self.image_info.append(info)
            
        except Exception as e:
            self.corrupt_files.append({'path': str(img_path), 'error': str(e)})
    
    def analyze_class_distribution(self):
        """Analyze and visualize class distribution."""
        print("\n" + "="*60)
        print("Class Distribution Analysis")
        print("="*60)
        
        df = pd.DataFrame(self.image_info)
        
        # Overall distribution
        print("\nOverall class distribution:")
        print(df['class'].value_counts())
        print(f"\nClass balance ratio: {df['class'].value_counts().min() / df['class'].value_counts().max():.3f}")
        
        # Per-split distribution
        print("\nPer-split distribution:")
        split_class = df.groupby(['split', 'class']).size().unstack(fill_value=0)
        print(split_class)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Overall distribution
        df['class'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
        axes[0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        
        # Per-split distribution
        split_class.T.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].legend(title='Split')
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_path / 'class_distribution.png'}")
        plt.close()
    
    def analyze_image_properties(self):
        """Analyze image dimensions and properties."""
        print("\n" + "="*60)
        print("Image Properties Analysis")
        print("="*60)
        
        df = pd.DataFrame(self.image_info)
        
        # Dimensions
        print("\nImage dimensions:")
        print(f"Width:  min={df['width'].min()}, max={df['width'].max()}, mean={df['width'].mean():.1f}")
        print(f"Height: min={df['height'].min()}, max={df['height'].max()}, mean={df['height'].mean():.1f}")
        print(f"Aspect ratio: min={df['aspect_ratio'].min():.3f}, max={df['aspect_ratio'].max():.3f}, mean={df['aspect_ratio'].mean():.3f}")
        
        # Color modes
        print("\nColor modes:")
        print(df['mode'].value_counts())
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Width distribution
        axes[0, 0].hist(df['width'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Width Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['width'].mean(), color='red', linestyle='--', label=f'Mean: {df["width"].mean():.0f}')
        axes[0, 0].legend()
        
        # Height distribution
        axes[0, 1].hist(df['height'], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Image Height Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(df['height'].mean(), color='red', linestyle='--', label=f'Mean: {df["height"].mean():.0f}')
        axes[0, 1].legend()
        
        # Aspect ratio
        axes[1, 0].hist(df['aspect_ratio'], bins=50, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['aspect_ratio'].mean(), color='red', linestyle='--', label=f'Mean: {df["aspect_ratio"].mean():.3f}')
        axes[1, 0].legend()
        
        # File size
        axes[1, 1].hist(df['file_size'] / 1024, bins=50, color='plum', edgecolor='black')
        axes[1, 1].set_title('File Size Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('File Size (KB)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline((df['file_size'] / 1024).mean(), color='red', linestyle='--', label=f'Mean: {(df["file_size"] / 1024).mean():.0f} KB')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'image_properties.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_path / 'image_properties.png'}")
        plt.close()
    
    def analyze_pixel_intensities(self):
        """Analyze pixel intensity distributions."""
        print("\n" + "="*60)
        print("Pixel Intensity Analysis")
        print("="*60)
        
        df = pd.DataFrame(self.image_info)
        
        print("\nIntensity statistics:")
        print(f"Mean: {df['mean_intensity'].mean():.2f} ± {df['mean_intensity'].std():.2f}")
        print(f"Range: [{df['min_intensity'].min():.0f}, {df['max_intensity'].max():.0f}]")
        
        # Per-class analysis
        print("\nPer-class intensity:")
        for class_name in df['class'].unique():
            class_df = df[df['class'] == class_name]
            print(f"  {class_name}: {class_df['mean_intensity'].mean():.2f} ± {class_df['mean_intensity'].std():.2f}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Mean intensity distribution
        for class_name in df['class'].unique():
            class_df = df[df['class'] == class_name]
            axes[0].hist(class_df['mean_intensity'], bins=50, alpha=0.6, label=class_name, edgecolor='black')
        axes[0].set_title('Mean Intensity Distribution by Class', fontweight='bold')
        axes[0].set_xlabel('Mean Pixel Intensity')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Standard deviation distribution
        for class_name in df['class'].unique():
            class_df = df[df['class'] == class_name]
            axes[1].hist(class_df['std_intensity'], bins=50, alpha=0.6, label=class_name, edgecolor='black')
        axes[1].set_title('Intensity Std Dev Distribution by Class', fontweight='bold')
        axes[1].set_xlabel('Std Dev of Pixel Intensity')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'pixel_intensities.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_path / 'pixel_intensities.png'}")
        plt.close()
    
    def check_data_quality(self):
        """Check for corrupt files."""
        print("\n" + "="*60)
        print("Data Quality Check")
        print("="*60)
        
        # Corrupt files
        print(f"\nCorrupt/unreadable files: {len(self.corrupt_files)}")
        if self.corrupt_files:
            print("First few corrupt files:")
            for corrupt in self.corrupt_files[:5]:
                print(f"  - {corrupt['path']}: {corrupt['error']}")
        else:
            print("✓ No corrupt files detected")
        
        # Save quality report
        quality_report = {
            'total_images': len(self.image_info),
            'corrupt_files': len(self.corrupt_files),
            'corrupt_details': self.corrupt_files[:10],
        }
        
        with open('reports/metrics/data_quality_report.json', 'w') as f:
            json.dump(quality_report, f, indent=2)
        print(f"\n✓ Saved: reports/metrics/data_quality_report.json")
    
    def visualize_sample_images(self, n_samples=8):
        """Visualize sample images from each class."""
        print("\n" + "="*60)
        print("Creating Sample Visualizations")
        print("="*60)
        
        df = pd.DataFrame(self.image_info)
        
        for class_name in df['class'].unique():
            class_df = df[df['class'] == class_name]
            samples = class_df.sample(min(n_samples, len(class_df)))
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for idx, (_, row) in enumerate(samples.iterrows()):
                img = Image.open(row['path'])
                axes[idx].imshow(img, cmap='gray')
                axes[idx].axis('off')
                axes[idx].set_title(f"{row['width']}x{row['height']}", fontsize=10)
            
            plt.suptitle(f'Sample Images: {class_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_path / f'samples_{class_name.lower()}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_path / f'samples_{class_name.lower()}.png'}")
            plt.close()
    
    def save_summary_stats(self):
        """Save summary statistics to CSV."""
        df = pd.DataFrame(self.image_info)
        
        # Overall summary
        summary = df.describe()
        summary.to_csv('reports/metrics/eda_summary_stats.csv')
        print(f"\n✓ Saved: reports/metrics/eda_summary_stats.csv")
        
        # Per-class summary
        class_summary = df.groupby('class').agg({
            'width': ['mean', 'std', 'min', 'max'],
            'height': ['mean', 'std', 'min', 'max'],
            'mean_intensity': ['mean', 'std'],
            'file_size': ['mean', 'std'],
        })
        class_summary.to_csv('reports/metrics/eda_class_summary.csv')
        print(f"✓ Saved: reports/metrics/eda_class_summary.csv")
    
    def run_full_eda(self):
        """Run complete EDA pipeline."""
        print("\n" + "="*70)
        print(" " * 20 + "CHEST X-RAY EDA")
        print("="*70)
        
        # Scan dataset
        self.scan_dataset()
        
        if not self.image_info:
            print("\n✗ No images found. Please check data path.")
            return
        
        # Run analyses
        self.analyze_class_distribution()
        self.analyze_image_properties()
        self.analyze_pixel_intensities()
        self.check_data_quality()
        self.visualize_sample_images()
        self.save_summary_stats()
        
        print("\n" + "="*70)
        print(" " * 25 + "EDA COMPLETE!")
        print("="*70)
        print("\nResults saved to:")
        print(f"  - Figures: {self.output_path}")
        print(f"  - Metrics: reports/metrics/")


def main():
    """Main entry point."""
    eda = ChestXRayEDA()
    eda.run_full_eda()


if __name__ == "__main__":
    main()
