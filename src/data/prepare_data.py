"""
Data preparation and preprocessing pipeline.

Creates stratified train/val/test splits with proper class distribution.
Saves split information for reproducible experiments.
"""

import os
import sys
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import set_seed, ensure_dir, load_config


def find_dataset_root(data_path):
    """Find the actual dataset root directory."""
    data_path = Path(data_path)
    possible_roots = [
        data_path,
        data_path / "chest_xray",
        data_path / "ChestXRay2017",
    ]
    
    for root in possible_roots:
        if root.exists():
            subdirs = [d.name.lower() for d in root.iterdir() if d.is_dir()]
            if any(split in subdirs for split in ['train', 'test', 'val']):
                return root
    return None


def collect_all_images(data_path):
    """
    Collect all image paths and labels from the dataset.
    
    Returns:
        DataFrame with columns: path, class, original_split
    """
    print("\n" + "="*60)
    print("Collecting All Images")
    print("="*60)
    
    dataset_root = find_dataset_root(data_path)
    if not dataset_root:
        raise ValueError(f"Could not find dataset in {data_path}")
    
    print(f"Dataset root: {dataset_root}")
    
    all_images = []
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_root / split
        if not split_path.exists():
            split_path = dataset_root / split.upper()
        if not split_path.exists():
            print(f"⚠ Split '{split}' not found, skipping")
            continue
        
        print(f"\nScanning {split}/...")
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_files = (
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.png"))
            )
            
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_path in image_files:
                all_images.append({
                    'path': str(img_path),
                    'class': class_name,
                    'original_split': split
                })
    
    df = pd.DataFrame(all_images)
    print(f"\n✓ Collected {len(df)} total images")
    print(f"  Classes: {df['class'].unique().tolist()}")
    print(f"  Class distribution:\n{df['class'].value_counts()}")
    
    return df


def create_stratified_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create stratified train/val/test splits.
    
    Args:
        df: DataFrame with image paths and labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    print("\n" + "="*60)
    print("Creating Stratified Splits")
    print("="*60)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    set_seed(random_seed)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['class'],
        random_state=random_seed
    )
    
    # Second split: separate validation from training
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['class'],
        random_state=random_seed
    )
    
    # Report
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nClass distribution in splits:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n  {split_name}:")
        for class_name in sorted(split_df['class'].unique()):
            count = (split_df['class'] == class_name).sum()
            pct = count / len(split_df) * 100
            print(f"    {class_name}: {count} ({pct:.1f}%)")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir="data/processed"):
    """
    Save split information to disk.
    
    Args:
        train_df, val_df, test_df: DataFrames with split information
        output_dir: Directory to save splits
    """
    print("\n" + "="*60)
    print("Saving Split Information")
    print("="*60)
    
    ensure_dir(output_dir)
    
    # Save as CSV
    train_df.to_csv(f"{output_dir}/train_split.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_split.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_split.csv", index=False)
    
    print(f"✓ Saved splits to {output_dir}/")
    
    # Save split summary as JSON
    summary = {
        'total_images': len(train_df) + len(val_df) + len(test_df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'classes': sorted(train_df['class'].unique().tolist()),
        'class_distribution': {
            'train': train_df['class'].value_counts().to_dict(),
            'val': val_df['class'].value_counts().to_dict(),
            'test': test_df['class'].value_counts().to_dict(),
        },
        'random_seed': 42,
    }
    
    with open(f"{output_dir}/split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {output_dir}/split_summary.json")
    
    return summary


def compute_class_weights(train_df):
    """
    Compute class weights for handling imbalance.
    
    Args:
        train_df: Training split DataFrame
        
    Returns:
        Dictionary mapping class names to weights
    """
    print("\n" + "="*60)
    print("Computing Class Weights")
    print("="*60)
    
    class_counts = train_df['class'].value_counts()
    total = len(train_df)
    n_classes = len(class_counts)
    
    # Inverse frequency weighting
    weights = {}
    for class_name, count in class_counts.items():
        weight = total / (n_classes * count)
        weights[class_name] = float(weight)
        print(f"  {class_name}: {weight:.4f} (count: {count})")
    
    # Save weights
    with open("data/processed/class_weights.json", 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"\n✓ Saved class weights to data/processed/class_weights.json")
    
    return weights


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*70)
    print(" " * 20 + "DATA PREPARATION")
    print("="*70)
    
    # Load config
    try:
        config = load_config()
        data_config = config.get('data', {})
        random_seed = config.get('seed', 42)
    except:
        print("⚠ Using default configuration")
        data_config = {}
        random_seed = 42
    
    # Set seed
    set_seed(random_seed)
    
    # Collect all images
    raw_data_path = data_config.get('raw_dir', 'data/raw')
    df = collect_all_images(raw_data_path)
    
    # Create splits
    train_ratio = data_config.get('train_split', 0.7)
    val_ratio = data_config.get('val_split', 0.15)
    test_ratio = data_config.get('test_split', 0.15)
    
    train_df, val_df, test_df = create_stratified_splits(
        df, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Save splits
    processed_dir = data_config.get('processed_dir', 'data/processed')
    summary = save_splits(train_df, val_df, test_df, processed_dir)
    
    # Compute class weights
    weights = compute_class_weights(train_df)
    
    print("\n" + "="*70)
    print(" " * 20 + "PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nSplit files saved to: {processed_dir}/")
    print(f"  - train_split.csv ({len(train_df)} images)")
    print(f"  - val_split.csv ({len(val_df)} images)")
    print(f"  - test_split.csv ({len(test_df)} images)")
    print(f"  - split_summary.json")
    print(f"  - class_weights.json")


if __name__ == "__main__":
    main()
