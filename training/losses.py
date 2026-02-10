import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_physgaze_losses(predictions: dict, targets: torch.Tensor, 
                           input_images: torch.Tensor, 
                           lambda_reg: float = 1.0,
                           lambda_cycle: float = 0.2,
                           lambda_acm: float = 0.1) -> dict:
    """
    Compute PhysGaze losses (Equation 6 in paper)
    
    Args:
        predictions: Dictionary containing gaze_init, gaze_corrected, rendered_img
        targets: Ground truth gaze angles
        input_images: Input images for cycle consistency
        lambda_reg: Weight for regression loss
        lambda_cycle: Weight for cycle consistency loss
        lambda_acm: Weight for ACM regularization loss
    
    Returns:
        Dictionary of losses
    """
    gaze_init = predictions['gaze_init']
    gaze_corrected = predictions['gaze_corrected']
    rendered_img = predictions['rendered_img']
    
    # 1. Regression loss (MAE)
    reg_loss = F.l1_loss(gaze_corrected, targets)
    
    # 2. Cycle-consistency loss (L1)
    cycle_loss = F.l1_loss(rendered_img, input_images)
    
    # 3. ACM regularization loss
    acm_loss = F.mse_loss(gaze_corrected, gaze_init) * 0.05
    
    # Total loss (Equation 6)
    total_loss = (lambda_reg * reg_loss +
                 lambda_cycle * cycle_loss +
                 lambda_acm * acm_loss)
    
    return {
        'total': total_loss,
        'reg': reg_loss,
        'cycle': cycle_loss,
        'acm': acm_loss
    }


def angular_error_loss(predictions: torch.Tensor, 
                      targets: torch.Tensor) -> torch.Tensor:
    """Compute angular error loss"""
    # Convert to vectors
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
    
    return angular_error.mean()
