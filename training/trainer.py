import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict

from models.physgaze import PhysGaze
from training.losses import PhysGazeLoss


class PhysGazeTrainer:
    """
    Training pipeline for PhysGaze model.
    """
    
    def __init__(
        self,
        model: PhysGaze,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_reg: float = 1.0,
        lambda_cycle: float = 0.2,
        lambda_acm: float = 0.1,
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        self.criterion = PhysGazeLoss(
            lambda_reg=lambda_reg,
            lambda_cycle=lambda_cycle,
            lambda_acm=lambda_acm
        )
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'outlier_rate': []
        }
        
        self.best_val_mae = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images, return_all=True)
            losses = self.criterion(outputs, targets, images)
            
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            total_mae += losses['angular_error'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mae': f"{losses['angular_error'].item():.2f}°"
            })
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        outliers = 0
        total_samples = 0
        
        for images, targets in tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]'):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images, return_all=False)
            losses = self.criterion(outputs, targets)
            
            if hasattr(self.model, 'acm'):
                valid = self.model.acm.is_anatomically_valid(outputs['gaze'])
                outliers += (~valid).sum().item()
            
            total_loss += losses['total'].item()
            total_mae += losses['angular_error'].item()
            num_batches += 1
            total_samples += images.shape[0]
        
        outlier_rate = (outliers / total_samples) * 100 if total_samples > 0 else 0
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'outlier_rate': outlier_rate
        }
    
    def train(self, num_epochs: int = 50, save_best: bool = True):
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['outlier_rate'].append(val_metrics['outlier_rate'])
            
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('MAE/train', train_metrics['mae'], epoch)
            self.writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
            self.writer.add_scalar('Outlier_Rate', val_metrics['outlier_rate'], epoch)
            self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}°")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}°, Outliers: {val_metrics['outlier_rate']:.1f}%")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            if save_best and val_metrics['mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['mae']
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ★ New best model saved! (MAE: {self.best_val_mae:.2f}°)")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation MAE: {self.best_val_mae:.2f}°")
        print("=" * 60)
        
        self.save_checkpoint(num_epochs - 1, 'final_model.pt')
        return self.history
    
    def save_checkpoint(self, epoch: int, filename: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'history': self.history
        }
        torch.save(checkpoint, self.log_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_mae = checkpoint['best_val_mae']
        self.history = checkpoint['history']
        return checkpoint['epoch']