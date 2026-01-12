"""
Loss functions for chest X-ray classification.

Implements focal loss and weighted cross-entropy for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        reduction: 'mean', 'sum' or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with class weights.
    
    Args:
        class_weights: List or tensor of weights for each class
    """
    
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        if isinstance(class_weights, list):
            class_weights = torch.FloatTensor(class_weights)
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights)


def create_criterion(config, class_weights=None, device='cuda'):
    """
    Create loss function from configuration.
    
    Args:
        config: Configuration dictionary
        class_weights: Class weights for weighted loss
        device: Device to put weights on
        
    Returns:
        criterion: Loss function
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'focal_loss')
    
    if loss_type == 'focal_loss':
        alpha = loss_config.get('focal_alpha', 0.25)
        gamma = loss_config.get('focal_gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"✓ Using Focal Loss (alpha={alpha}, gamma={gamma})")
        
    elif loss_type == 'weighted_cross_entropy':
        if class_weights is None:
            raise ValueError("Class weights required for weighted cross-entropy")
        if isinstance(class_weights, dict):
            # Convert dict to list in correct order
            sorted_classes = sorted(class_weights.keys())
            class_weights = [class_weights[cls] for cls in sorted_classes]
        criterion = WeightedCrossEntropyLoss(class_weights)
        criterion = criterion.to(device)
        print(f"✓ Using Weighted Cross-Entropy (weights={class_weights})")
        
    elif loss_type == 'cross_entropy':
        criterion = nn.CrossEntropy()
        print(f"✓ Using Cross-Entropy Loss")
        
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return criterion
