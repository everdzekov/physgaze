import torch
import numpy as np
from typing import Tuple


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_gaze(gaze: torch.Tensor) -> torch.Tensor:
    """Convert normalized gaze [-1, 1] to degrees."""
    yaw_max = 55.0
    pitch_max = 40.0
    
    gaze_deg = gaze.clone()
    gaze_deg[:, 0] = gaze_deg[:, 0] * yaw_max
    gaze_deg[:, 1] = gaze_deg[:, 1] * pitch_max
    
    return gaze_deg


def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate angular error in degrees."""
    pred_deg = denormalize_gaze(pred)
    target_deg = denormalize_gaze(target)
    
    error = torch.abs(pred_deg - target_deg)
    angular = torch.sqrt(error[:, 0]**2 + error[:, 1]**2)
    
    return angular.mean()