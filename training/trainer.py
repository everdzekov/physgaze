import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..models.physgaze import PhysGaze
from .dynamic_epoch import DynamicEpochManager

class EnhancedPhysGazeTrainer:
    """Enhanced trainer with dynamic epoch stopping"""
    
    def __init__(self, model: PhysGaze, train_loader: DataLoader, 
                 val_loader: DataLoader, device: torch.device, config: dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Dynamic epoch manager
        self.epoch_manager = DynamicEpochManager(config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 5),
            min_lr=1e-6
        )
        
        # History tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_outliers': [], 'val_outliers': [],
            'lr': []
        }
        
        self.best_val_mae = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_outliers = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(images, return_all=True)
            gaze_pred = predictions['gaze_corrected']
            
            # Compute losses
            losses = self.model.compute_losses(predictions, targets, images)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mae = nn.L1Loss()(gaze_pred, targets).item()
                outliers = self.model.get_outlier_rate(gaze_pred)
                
                batch_size = images.size(0)
                total_loss += losses['total'].item() * batch_size
                total_mae += mae * batch_size
                total_outliers += outliers * batch_size
                total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mae': f"{mae:.2f}°",
                'outliers': f"{outliers:.1f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        avg_outliers = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
        
        return avg_loss, avg_mae, avg_outliers
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, float, List, List, List]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_outliers = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_inits = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            predictions = self.model(images, return_all=True)
            gaze_pred = predictions['gaze_corrected']
            
            losses = self.model.compute_losses(predictions, targets, images)
            mae = nn.L1Loss()(gaze_pred, targets).item()
            outliers = self.model.get_outlier_rate(gaze_pred)
            
            batch_size = images.size(0)
            total_loss += losses['total'].item() * batch_size
            total_mae += mae * batch_size
            total_outliers += outliers * batch_size
            total_samples += batch_size
            
            all_preds.append(gaze_pred.cpu())
            all_targets.append(targets.cpu())
            all_inits.append(predictions['gaze_init'].cpu())
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mae': f"{mae:.2f}°",
                'outliers': f"{outliers:.1f}%"
            })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        avg_outliers = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
        
        self.scheduler.step(avg_mae)
        
        return avg_loss, avg_mae, avg_outliers, all_preds, all_targets, all_inits
    
    def train(self, max_epochs: int = 100) -> Dict:
        """Train with dynamic epoch stopping"""
        print(f"\n🚀 Starting training with dynamic epoch stopping...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print(f"Max epochs: {max_epochs}")
        print(f"Early stopping patience: {self.epoch_manager.patience}")
        
        for epoch in range(max_epochs):
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_mae, train_outliers = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_mae, val_outliers, _, _, _ = self.validate(epoch)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_outliers'].append(train_outliers)
            self.history['val_outliers'].append(val_outliers)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Update epoch manager
            should_stop = self.epoch_manager.update(
                train_loss, val_loss, train_mae, val_mae
            )
            
            # Print epoch summary
            print(f"\n📈 Epoch {epoch + 1} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}°, Outliers: {train_outliers:.1f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}°, Outliers: {val_outliers:.1f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate improvement
            if epoch > 0:
                val_improvement = ((self.history['val_mae'][-2] - val_mae) / 
                                  self.history['val_mae'][-2] * 100)
                print(f"  Val MAE Improvement: {val_improvement:+.2f}%")
            
            # Save best model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch + 1
                print(f"  🎯 New best model! (MAE: {val_mae:.2f}°)")
            
            # Check if we should stop
            if should_stop:
                print(f"\n🛑 Stopping early after {epoch + 1} epochs due to no improvement")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Loaded best model from epoch {self.best_epoch} (MAE: {self.best_val_mae:.2f}°)")
        
        # Get training summary
        summary = self.epoch_manager.get_training_summary()
        if summary:
            print(f"\n📋 Training Summary:")
            print(f"  Total epochs: {summary['epochs']}")
            print(f"  Best epoch: {self.best_epoch}")
            print(f"  Best val MAE: {self.best_val_mae:.2f}°")
            print(f"  Final train-val gap: {summary['train_val_gap']:.4f}")
            print(f"  Overfitting ratio: {summary['overfitting_ratio']:.2%}")
        
        return self.history
