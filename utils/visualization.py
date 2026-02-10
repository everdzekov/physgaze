import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Optional, List

class EnhancedVisualizer:
    """Enhanced visualization tools for gaze estimation"""
    
    @staticmethod
    def plot_training_curves(history: dict, title: str = "Training Curves", 
                            save_path: Optional[str] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE curves
        axes[0, 1].plot(epochs, history['train_mae'], 'g-', label='Train MAE', linewidth=2)
        axes[0, 1].plot(epochs, history['val_mae'], 'm-', label='Val MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (°)')
        axes[0, 1].set_title('MAE Progression')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Outlier rate
        if 'train_outliers' in history:
            axes[1, 0].plot(epochs, history['train_outliers'], 'c-', label='Train Outliers', linewidth=2)
            axes[1, 0].plot(epochs, history['val_outliers'], 'y-', label='Val Outliers', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Outlier Rate (%)')
            axes[1, 0].set_title('Outlier Reduction')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved training curves to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_gaze_comparison(predictions: np.ndarray, targets: np.ndarray, 
                            initial_predictions: Optional[np.ndarray] = None,
                            title: str = "Gaze Comparison"):
        """Visualize gaze predictions vs targets"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Scatter plot
        axes[0].scatter(targets[:, 0], targets[:, 1], alpha=0.5, 
                       label='Ground Truth', s=10)
        axes[0].scatter(predictions[:, 0], predictions[:, 1], alpha=0.5, 
                       label='Predictions', s=10)
        if initial_predictions is not None:
            axes[0].scatter(initial_predictions[:, 0], initial_predictions[:, 1], 
                          alpha=0.3, label='Initial Preds', s=5)
        
        axes[0].set_xlabel('Yaw (degrees)')
        axes[0].set_ylabel('Pitch (degrees)')
        axes[0].set_title('Gaze Angle Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add physiological limits
        yaw_limit = 50
        pitch_limit = 35
        axes[0].axvline(x=-yaw_limit, color='r', linestyle='--', alpha=0.3)
        axes[0].axvline(x=yaw_limit, color='r', linestyle='--', alpha=0.3)
        axes[0].axhline(y=-pitch_limit, color='r', linestyle='--', alpha=0.3)
        axes[0].axhline(y=pitch_limit, color='r', linestyle='--', alpha=0.3)
        
        # Error histogram
        errors = np.sqrt((predictions[:, 0] - targets[:, 0])**2 + 
                        (predictions[:, 1] - targets[:, 1])**2)
        axes[1].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1].axvline(np.mean(errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(errors):.2f}°')
        axes[1].set_xlabel('Angular Error (degrees)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Error vs magnitude
        target_magnitudes = np.sqrt(targets[:, 0]**2 + targets[:, 1]**2)
        axes[2].scatter(target_magnitudes, errors, alpha=0.5, s=10)
        axes[2].set_xlabel('Gaze Angle Magnitude (degrees)')
        axes[2].set_ylabel('Prediction Error (degrees)')
        axes[2].set_title('Error vs Gaze Magnitude')
        
        if len(target_magnitudes) > 1:
            z = np.polyfit(target_magnitudes, errors, 1)
            p = np.poly1d(z)
            axes[2].plot(sorted(target_magnitudes), p(sorted(target_magnitudes)), 
                        'r--', label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_loso_results(subjects: List[str], maes: List[float],
                         training_epochs: List[int], save_path: Optional[str] = None):
        """Plot LOSO cross-validation results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE per subject
        x_pos = np.arange(len(subjects))
        axes[0].bar(x_pos, maes, alpha=0.7, color='steelblue')
        axes[0].axhline(y=np.mean(maes), color='red', linestyle='--',
                       label=f'Mean: {np.mean(maes):.2f}°')
        axes[0].set_xlabel('Subject')
        axes[0].set_ylabel('MAE (°)')
        axes[0].set_title('MAE per Subject (LOSO)')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(subjects, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training epochs per subject
        axes[1].bar(x_pos, training_epochs, alpha=0.7, color='orange')
        axes[1].axhline(y=np.mean(training_epochs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(training_epochs):.1f} epochs')
        axes[1].set_xlabel('Subject')
        axes[1].set_ylabel('Training Epochs')
        axes[1].set_title('Training Efficiency per Subject')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(subjects, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
