import torch
import torch.nn.functional as F
import numpy as np

def compute_gaze_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute comprehensive gaze estimation metrics
    """
    # Basic MAE
    mae = F.l1_loss(predictions, targets).item()
    mae_yaw = F.l1_loss(predictions[:, 0], targets[:, 0]).item()
    mae_pitch = F.l1_loss(predictions[:, 1], targets[:, 1]).item()
    
    # MSE and RMSE
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse).item()
    
    # Angular error
    angular_errors = compute_angular_error(predictions, targets)
    mean_angular_error = torch.mean(angular_errors).item()
    std_angular_error = torch.std(angular_errors).item()
    
    # Accuracy at different thresholds
    thresholds = [5.0, 10.0, 15.0]
    accuracies = {}
    for thresh in thresholds:
        acc = (angular_errors <= thresh).float().mean().item() * 100
        accuracies[f'acc_{int(thresh)}'] = acc
    
    return {
        'mae': mae,
        'mae_yaw': mae_yaw,
        'mae_pitch': mae_pitch,
        'mse': mse.item(),
        'rmse': rmse,
        'mean_angular_error': mean_angular_error,
        'std_angular_error': std_angular_error,
        'angular_errors': angular_errors,
        **accuracies
    }

def compute_angular_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute angular error between predicted and target gaze vectors"""
    # Convert angles to vectors
    pred_yaw = torch.deg2rad(predictions[:, 0])
    pred_pitch = torch.deg2rad(predictions[:, 1])
    
    target_yaw = torch.deg2rad(targets[:, 0])
    target_pitch = torch.deg2rad(targets[:, 1])
    
    # Compute vectors
    pred_x = torch.sin(pred_yaw) * torch.cos(pred_pitch)
    pred_y = torch.sin(pred_pitch)
    pred_z = torch.cos(pred_yaw) * torch.cos(pred_pitch)
    
    target_x = torch.sin(target_yaw) * torch.cos(target_pitch)
    target_y = torch.sin(target_pitch)
    target_z = torch.cos(target_yaw) * torch.cos(target_pitch)
    
    pred_vec = torch.stack([pred_x, pred_y, pred_z], dim=1)
    target_vec = torch.stack([target_x, target_y, target_z], dim=1)
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    # Compute angular error in degrees
    angular_error = torch.rad2deg(torch.acos(cos_sim))
    
    return angular_error

def compute_outlier_rate(predictions: torch.Tensor, 
                        yaw_limit: float = 50.0, 
                        pitch_limit: float = 35.0) -> float:
    """Compute percentage of physiologically impossible predictions"""
    yaw_outliers = torch.abs(predictions[:, 0]) > yaw_limit
    pitch_outliers = torch.abs(predictions[:, 1]) > pitch_limit
    outliers = torch.logical_or(yaw_outliers, pitch_outliers)
    return outliers.float().mean().item() * 100
