import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

from models.physgaze import PhysGaze


class PhysGazeEvaluator:
    """
    Evaluation and visualization utilities for PhysGaze.
    """
    
    def __init__(self, model: PhysGaze, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.yaw_max = 55.0
        self.pitch_max = 40.0
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        all_preds = []
        all_targets = []
        all_corrections = []
        
        for images, targets in tqdm(test_loader, desc='Evaluating'):
            images = images.to(self.device)
            outputs = self.model(images, return_all=False)
            
            all_preds.append(outputs['gaze'].cpu())
            all_targets.append(targets)
            all_corrections.append(outputs['correction'].cpu())
        
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        corrections = torch.cat(all_corrections, dim=0).numpy()
        
        preds_deg = preds.copy()
        preds_deg[:, 0] *= self.yaw_max
        preds_deg[:, 1] *= self.pitch_max
        
        targets_deg = targets.copy()
        targets_deg[:, 0] *= self.yaw_max
        targets_deg[:, 1] *= self.pitch_max
        
        errors = np.abs(preds_deg - targets_deg)
        angular_errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)
        
        outlier_yaw = 60.0
        outlier_pitch = 45.0
        outliers = (np.abs(preds_deg[:, 0]) > outlier_yaw) | (np.abs(preds_deg[:, 1]) > outlier_pitch)
        
        extreme_mask = (np.abs(targets_deg[:, 0]) > 40) | (np.abs(targets_deg[:, 1]) > 30)
        
        metrics = {
            'mae': np.mean(angular_errors),
            'mae_yaw': np.mean(errors[:, 0]),
            'mae_pitch': np.mean(errors[:, 1]),
            'std': np.std(angular_errors),
            'median': np.median(angular_errors),
            'p95': np.percentile(angular_errors, 95),
            'outlier_rate': np.mean(outliers) * 100,
            'extreme_pose_mae': np.mean(angular_errors[extreme_mask]) if np.any(extreme_mask) else 0,
            'normal_pose_mae': np.mean(angular_errors[~extreme_mask]) if np.any(~extreme_mask) else 0,
            'mean_correction': np.mean(np.abs(corrections))
        }
        
        return metrics, preds_deg, targets_deg, angular_errors