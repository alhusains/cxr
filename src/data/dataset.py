"""
PyTorch Dataset and DataLoader for Chest X-Ray images.

Implements medical-appropriate preprocessing and augmentation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import set_seed, load_config


class ChestXRayDataset(Dataset):
    """
    Dataset for chest X-ray images.
    
    Args:
        csv_file: Path to CSV file with image paths and labels
        transform: Albumentations transform pipeline
        grayscale: Whether to convert images to grayscale
    """
    
    def __init__(self, csv_file, transform=None, grayscale=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.grayscale = grayscale
        
        # Create label mapping
        self.classes = sorted(self.data['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.data)} images from {csv_file}")
        print(f"Classes: {self.classes}")
        print(f"Class distribution: {dict(self.data['class'].value_counts())}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.data.iloc[idx]['path']
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale if needed
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Convert back to 3-channel for compatibility with ImageNet models
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        class_name = self.data.iloc[idx]['class']
        label = self.class_to_idx[class_name]
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name from index."""
        return self.idx_to_class[idx]


def get_transforms(config, is_training=True):
    """
    Get albumentations transform pipeline.
    
    Args:
        config: Configuration dictionary
        is_training: Whether this is for training (with augmentation)
        
    Returns:
        Albumentations Compose object
    """
    preprocess_config = config.get('preprocessing', {})
    aug_config = config.get('augmentation', {})
    
    target_size = preprocess_config.get('target_size', [224, 224])
    mean = preprocess_config.get('normalize_mean', [0.485, 0.456, 0.406])
    std = preprocess_config.get('normalize_std', [0.229, 0.224, 0.225])
    
    # Strategy: Resize with aspect ratio preservation + padding
    # Justification: Maintains anatomical proportions; padding doesn't distort lung structure
    # Medical X-rays don't have fixed regions of interest, but anatomy proportions matter
    use_clahe = preprocess_config.get('use_clahe', False)
    
    if is_training:
        # Training transforms with medical-appropriate augmentation
        transform_list = [
            # CLAHE for contrast enhancement (optional, can improve under/over-exposed images)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3) if use_clahe else A.NoOp(),
            
            # Medical-appropriate augmentations (before resize to preserve detail)
            A.HorizontalFlip(p=0.5) if aug_config.get('horizontal_flip', True) else A.NoOp(),
            
            # Combined affine transform (rotation, translation, scale)
            A.Affine(
                rotate=(-aug_config.get('rotation_degrees', 10), aug_config.get('rotation_degrees', 10)),
                translate_percent={
                    'x': (-aug_config.get('translation', 0.05), aug_config.get('translation', 0.05)),
                    'y': (-aug_config.get('translation', 0.05), aug_config.get('translation', 0.05))
                },
                scale=(aug_config.get('scale', [0.95, 1.05])[0], aug_config.get('scale', [0.95, 1.05])[1]),
                p=0.6,
                mode=cv2.BORDER_CONSTANT,
                cval=0
            ),
            
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness', 0.1),
                contrast_limit=aug_config.get('contrast', 0.1),
                p=0.5
            ),
            
            A.GaussNoise(
                var_limit=(0, aug_config.get('gaussian_noise', 0.01)),
                mean=0,
                p=0.3
            ),
            
            # Resize: Longest side to target, then pad to square
            # This preserves aspect ratio while ensuring consistent output
            A.LongestMaxSize(max_size=target_size[0], always_apply=True),
            A.PadIfNeeded(
                min_height=target_size[0],
                min_width=target_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                always_apply=True
            ),
            
            # Normalization
            A.Normalize(mean=mean, std=std),
            
            # Convert to tensor
            ToTensorV2(),
        ]
    else:
        # Validation/test transforms (no augmentation, same resize strategy)
        transform_list = [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0) if use_clahe else A.NoOp(),
            A.LongestMaxSize(max_size=target_size[0], always_apply=True),
            A.PadIfNeeded(
                min_height=target_size[0],
                min_width=target_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                always_apply=True
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    
    return A.Compose(transform_list)


def create_dataloaders(config, train_csv, val_csv, test_csv):
    """
    Create PyTorch DataLoaders for train, val, and test sets.
    
    Args:
        config: Configuration dictionary
        train_csv: Path to training split CSV
        val_csv: Path to validation split CSV
        test_csv: Path to test split CSV
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    print("\n" + "="*60)
    print("Creating DataLoaders")
    print("="*60)
    
    # Get config values
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('compute', {}).get('num_workers', 4)
    pin_memory = config.get('compute', {}).get('pin_memory', True)
    
    # Create datasets
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    train_dataset = ChestXRayDataset(train_csv, transform=train_transform)
    val_dataset = ChestXRayDataset(val_csv, transform=val_transform)
    test_dataset = ChestXRayDataset(test_csv, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for training stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} images)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} images)")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def test_dataloader():
    """Test dataloader functionality."""
    print("\n" + "="*60)
    print("Testing DataLoader")
    print("="*60)
    
    config = load_config()
    set_seed(config.get('seed', 42))
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        config,
        train_csv="data/processed/train_split.csv",
        val_csv="data/processed/val_split.csv",
        test_csv="data/processed/test_split.csv"
    )
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch information:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels: {labels[:8].tolist()}")
    print(f"  Class names: {class_names}")
    
    # Check normalization
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    print(f"\nActual batch statistics:")
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std:  {std.tolist()}")
    
    print("\n✓ DataLoader test passed!")
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    test_dataloader()
