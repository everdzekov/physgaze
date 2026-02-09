import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class GazeTrainer:
    """Trainer for gaze estimation model."""
    
    def __init__(self, model, train_loader, val_loader, device, lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        # FIXED: Remove 'verbose' parameter for ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.criterion = nn.L1Loss()  # MAE loss
        
        self.best_val_mae = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # Calculate MAE in degrees
            outputs_deg = outputs * torch.tensor([30.0, 20.0]).to(self.device)
            targets_deg = targets * torch.tensor([30.0, 20.0]).to(self.device)
            mae = torch.mean(torch.abs(outputs_deg - targets_deg)).item()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{mae:.2f}°",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        return total_loss / total_samples, total_mae / total_samples
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_samples = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            
            loss = self.criterion(outputs, targets)
            outputs_deg = outputs * torch.tensor([30.0, 20.0]).to(self.device)
            targets_deg = targets * torch.tensor([30.0, 20.0]).to(self.device)
            mae = torch.mean(torch.abs(outputs_deg - targets_deg)).item()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae * batch_size
            total_samples += batch_size
        
        val_loss = total_loss / total_samples if total_samples > 0 else 0
        val_mae = total_mae / total_samples if total_samples > 0 else 0
        
        self.scheduler.step(val_mae)
        
        return val_loss, val_mae
    
    def train(self, epochs=30):
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_loss, train_mae = self.train_epoch(epoch)
            val_loss, val_mae = self.validate(epoch)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}°")
            print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}°")
            
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), 'best_mpiigaze_model.pth')
                print(f"  ★ New best model! (MAE: {val_mae:.2f}°)")
        
        print(f"\n✅ Training completed!")
        print(f"Best validation MAE: {self.best_val_mae:.2f}°")
        
        return self.history
