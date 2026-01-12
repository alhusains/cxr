"""
Model architectures for chest X-ray classification.

Supports multiple architectures with ImageNet pretraining.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChestXRayClassifier(nn.Module):
    """
    Chest X-ray classifier based on pretrained CNN backbones.
    
    Args:
        architecture: Model architecture (resnet50, densenet121, efficientnet_b0)
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(self, architecture='resnet50', num_classes=3, pretrained=True, dropout=0.5):
        super(ChestXRayClassifier, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Load backbone
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
            
        elif architecture == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        print(f"✓ Loaded {architecture} (pretrained={pretrained})")
        print(f"  Backbone features: {num_features}")
        print(f"  Output classes: {num_classes}")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def get_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: PyTorch model
    """
    model_config = config.get('model', {})
    
    architecture = model_config.get('architecture', 'resnet50')
    num_classes = model_config.get('num_classes', 3)
    pretrained = model_config.get('pretrained', True)
    dropout = model_config.get('dropout', 0.5)
    
    model = ChestXRayClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.utils import load_config
    
    config = load_config()
    
    print("\n" + "="*60)
    print("Testing Model Architecture")
    print("="*60)
    
    model = create_model(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    # Test freezing
    print("\nTesting backbone freezing...")
    model.freeze_backbone()
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    model.unfreeze_backbone()
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    print("\n✓ Model test passed!")
