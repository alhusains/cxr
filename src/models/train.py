"""
Complete training pipeline for chest X-ray classification.

Implements:
- DenseNet-121 with transfer learning
- Focal loss for class imbalance
- MLflow experiment tracking
- Early stopping and checkpointing
- Learning rate scheduling
- Comprehensive metrics
"""

import os
import sys
from pathlib import Path
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import mlflow
import mlflow.pytorch

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import set_seed, load_config, get_device, ensure_dir
from src.data.dataset import create_dataloaders
from src.models.model import create_model
from src.models.losses import create_criterion


class Trainer:
    """
    Complete training pipeline with experiment tracking.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration dictionary
        device: Device to train on
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, 
                 optimizer, scheduler, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training config
        self.epochs = config['training']['epochs']
        self.gradient_clip = config['training'].get('gradient_clip_value', 1.0)
        self.mixed_precision = config['training'].get('mixed_precision', True)
        
        # Early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 15)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config['logging']['model_registry'])
        ensure_dir(self.checkpoint_dir)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
            
            # Log batch metrics to MLflow
            if batch_idx % self.config['logging'].get('log_interval', 10) == 0:
                mlflow.log_metric('batch_loss', loss.item(), step=epoch * len(self.train_loader) + batch_idx)
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]  ")
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            mlflow.pytorch.log_model(self.model, "best_model")
    
    def train(self):
        """Complete training loop."""
        print("\n" + "="*70)
        print(" " * 25 + "TRAINING START")
        print("="*70)
        
        start_time = time.time()
        
        # Freeze backbone for initial epochs if configured
        freeze_epochs = self.config['model'].get('freeze_backbone_epochs', 0)
        if freeze_epochs > 0:
            print(f"\n🔒 Freezing backbone for first {freeze_epochs} epochs")
            self.model.freeze_backbone()
        
        for epoch in range(self.epochs):
            # Unfreeze after specified epochs
            if epoch == freeze_epochs and freeze_epochs > 0:
                print(f"\n🔓 Unfreezing backbone at epoch {epoch+1}")
                self.model.unfreeze_backbone()
                # Optionally adjust learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr,
                'generalization_gap': train_loss - val_loss
            }, step=epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f} | Gen Gap: {train_loss - val_loss:.4f}")
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  ✓ New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging'].get('save_checkpoint_interval', 5) == 0:
                self.save_checkpoint(epoch, is_best)
            elif is_best:
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
                break
        
        # Training complete
        training_time = time.time() - start_time
        print("\n" + "="*70)
        print(" " * 25 + "TRAINING COMPLETE")
        print("="*70)
        print(f"\nTotal training time: {training_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        
        # Save final checkpoint and history
        self.save_checkpoint(epoch, is_best=False)
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        mlflow.log_artifact(str(self.checkpoint_dir / 'training_history.json'))
        mlflow.log_metric('training_time_hours', training_time/3600)
        mlflow.log_metric('best_epoch', self.best_epoch)
        
        return self.history


def train_single_run(config, device, run_name="training_run"):
    """Train a single model with given config."""
    mlflow.set_experiment(config['logging']['experiment_name'])
    
    with mlflow.start_run(run_name=run_name):
        # Log configuration
        mlflow.log_params({
            'architecture': config['model']['architecture'],
            'pretrained': config['model']['pretrained'],
            'num_classes': config['model']['num_classes'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'optimizer': config['training']['optimizer'],
            'scheduler': config['training']['scheduler'],
            'loss_type': config['loss']['type'],
            'epochs': config['training']['epochs'],
            'seed': config['seed'],
            'use_clahe': config['preprocessing']['use_clahe'],
        })
        
        # Create dataloaders
        print("\n📊 Loading data...")
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            config,
            train_csv="data/processed/train_split.csv",
            val_csv="data/processed/val_split.csv",
            test_csv="data/processed/test_split.csv"
        )
        
        # Load class weights
        with open('data/processed/class_weights.json', 'r') as f:
            class_weights_dict = json.load(f)
        
        class_weights = torch.FloatTensor([class_weights_dict[c] for c in sorted(class_weights_dict.keys())])
        print(f"\n⚖️  Class weights: {dict(zip(sorted(class_weights_dict.keys()), class_weights.tolist()))}")
        
        # Create model
        print(f"\n🏗️  Building model...")
        model = create_model(config)
        model = model.to(device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        mlflow.log_param('total_parameters', total_params)
        mlflow.log_param('trainable_parameters', trainable_params)
        
        # Create loss function
        criterion = create_criterion(config, class_weights=class_weights_dict, device=device)
        
        # Create optimizer
        optimizer_name = config['training']['optimizer'].lower()
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        print(f"\n⚙️  Optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})")
        
        # Create scheduler
        scheduler_name = config['training']['scheduler']
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training']['scheduler_params']['T_max'],
                eta_min=config['training']['scheduler_params']['eta_min']
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        else:
            scheduler = None
        
        print(f"  Scheduler: {scheduler_name}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device
        )
        
        # Train
        history = trainer.train()
        
        print("\n✅ Training pipeline complete!")
        print(f"📁 Checkpoints saved to: {trainer.checkpoint_dir}")
        print(f"📊 MLflow run: {mlflow.active_run().info.run_id}")
        
        return trainer.best_val_loss, history


def hyperparameter_tuning(base_config, device, n_trials=15):
    """
    Perform Bayesian hyperparameter optimization using Optuna.
    
    Args:
        base_config: Base configuration dictionary
        device: Device to train on
        n_trials: Number of optimization trials
        
    Returns:
        best_params: Best hyperparameters found
        best_value: Best validation loss achieved
    """
    import optuna
    
    print("\n" + "="*70)
    print(" " * 15 + "HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("="*70)
    print(f"\nRunning Bayesian optimization with {n_trials} trials")
    print("This will take several hours...")
    
    def objective(trial):
        """Optuna objective function."""
        # Create DEEP copy of config to avoid modifying base_config
        trial_config = copy.deepcopy(base_config)
        
        # Suggest hyperparameters
        trial_config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        trial_config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        trial_config['model']['dropout'] = trial.suggest_float('dropout', 0.3, 0.7)
        trial_config['model']['freeze_backbone_epochs'] = trial.suggest_int('freeze_epochs', 0, 10)
        trial_config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        trial_config['loss']['focal_gamma'] = trial.suggest_float('focal_gamma', 1.5, 2.5)
        
        # Reduce epochs for tuning trials (early stopping will trigger earlier)
        trial_config['training']['epochs'] = 30
        trial_config['training']['early_stopping_patience'] = 8
        
        print(f"\n{'='*70}")
        print(f"Trial {trial.number + 1}/{n_trials}")
        print(f"  LR: {trial_config['training']['learning_rate']:.6f}")
        print(f"  Weight decay: {trial_config['training']['weight_decay']:.6f}")
        print(f"  Dropout: {trial_config['model']['dropout']:.2f}")
        print(f"  Freeze epochs: {trial_config['model']['freeze_backbone_epochs']}")
        print(f"  Batch size: {trial_config['training']['batch_size']}")
        print(f"  Focal gamma: {trial_config['loss']['focal_gamma']:.2f}")
        print(f"{'='*70}")
        
        # Train with trial config
        best_val_loss, _ = train_single_run(
            trial_config, 
            device, 
            run_name=f"optuna_trial_{trial.number+1}"
        )
        
        return best_val_loss
    
    # Create and run study
    study = optuna.create_study(
        direction='minimize',
        study_name='cxr_classification_tuning',
        sampler=optuna.samplers.TPESampler(seed=base_config['seed'])
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "="*70)
    print(" " * 15 + "HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\n✓ Best validation loss: {study.best_value:.4f}")
    print(f"\n✓ Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Create best config with DEEP copy
    best_config = copy.deepcopy(base_config)
    best_config['training']['learning_rate'] = study.best_params['learning_rate']
    best_config['training']['weight_decay'] = study.best_params['weight_decay']
    best_config['model']['dropout'] = study.best_params['dropout']
    best_config['model']['freeze_backbone_epochs'] = study.best_params['freeze_epochs']
    best_config['training']['batch_size'] = study.best_params['batch_size']
    best_config['loss']['focal_gamma'] = study.best_params['focal_gamma']
    
    # Full training epochs for final model
    best_config['training']['epochs'] = 100
    best_config['training']['early_stopping_patience'] = 15
    
    # Save to file
    ensure_dir('models')
    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_val_loss': float(study.best_value),
            'n_trials': n_trials
        }, f, indent=2)
    
    print(f"\n✓ Best hyperparameters saved to models/best_hyperparameters.json")
    
    return best_config, study.best_value


def main():
    """Main training function with optional hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train chest X-ray classification model')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning before final training')
    parser.add_argument('--n_trials', type=int, default=15,
                       help='Number of tuning trials (default: 15)')
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config)
    print(f"\nUsing device: {device}")
    
    if args.tune:
        # Hyperparameter tuning
        print("\n🔍 HYPERPARAMETER TUNING MODE")
        print(f"   Will run {args.n_trials} optimization trials")
        print(f"   Then train final model with best hyperparameters")
        
        best_config, best_val_loss = hyperparameter_tuning(config, device, args.n_trials)
        
        print("\n" + "="*70)
        print(" " * 15 + "TRAINING FINAL MODEL WITH BEST CONFIG")
        print("="*70)
        
        # Train final model with best hyperparameters
        final_val_loss, history = train_single_run(
            best_config, 
            device, 
            run_name="final_model_best_hyperparams"
        )
        
        print(f"\n✅ Final model validation loss: {final_val_loss:.4f}")
        print(f"   (Improvement from tuning: {best_val_loss - final_val_loss:.4f})")
        
    else:
        # Standard training with default config
        print("\n🚀 STANDARD TRAINING MODE")
        print("   Using default hyperparameters from config")
        print("   To enable hyperparameter tuning, use: python train.py --tune")
        
        train_single_run(config, device, run_name="training_default_config")


if __name__ == "__main__":
    main()
