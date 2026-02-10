import numpy as np
from typing import List, Optional, Dict

class DynamicEpochManager:
    """
    Manages dynamic epoch stopping based on overfitting detection
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.patience = config.get('early_stopping_patience', 10)
        self.min_delta = config.get('early_stopping_min_delta', 0.001)
        self.overfitting_threshold = config.get('overfitting_threshold', 0.1)
        
        # Track history
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
    
    def update(self, train_loss: float, val_loss: float, 
               train_mae: float, val_mae: float) -> bool:
        """Update metrics and check for overfitting"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_maes.append(train_mae)
        self.val_maes.append(val_mae)
        
        # Check for improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Check for overfitting
        if len(self.train_losses) > 5:
            recent_train_trend = self._calculate_trend(self.train_losses[-5:])
            recent_val_trend = self._calculate_trend(self.val_losses[-5:])
            
            # Overfitting detected when train loss is decreasing but val loss is increasing
            overfitting_detected = (recent_train_trend < -0.01 and recent_val_trend > 0.01 and 
                                   abs(recent_val_trend - recent_train_trend) > self.overfitting_threshold)
            
            if overfitting_detected:
                print(f"⚠️  Overfitting detected! Train trend: {recent_train_trend:.4f}, Val trend: {recent_val_trend:.4f}")
        
        # Check early stopping
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            print(f"🛑 Early stopping triggered after {len(self.train_losses)} epochs")
        
        return self.should_stop
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend of values"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeff = np.polyfit(x, values, 1)
        return coeff[0]  # Slope
    
    def get_training_summary(self) -> Optional[Dict]:
        """Get summary of training dynamics"""
        if len(self.train_losses) < 2:
            return None
            
        return {
            'epochs': len(self.train_losses),
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_mae': self.train_maes[-1],
            'final_val_mae': self.val_maes[-1],
            'best_val_loss': self.best_val_loss,
            'overfitting_ratio': (self.val_losses[-1] - self.train_losses[-1]) / self.train_losses[-1] if self.train_losses[-1] > 0 else 0,
            'train_val_gap': self.val_losses[-1] - self.train_losses[-1]
        }
