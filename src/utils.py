"""Utility functions for the CXR classification project"""

import random
import numpy as np
import torch
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(config: Dict[str, Any] = None) -> torch.device:
    """
    Get the device for training (CUDA if available).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device object
    """
    if config and config.get('compute', {}).get('device') == 'cpu':
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(directory: str):
    """
    Ensure that a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent
