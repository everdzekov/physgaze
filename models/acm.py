import torch
import torch.nn as nn
from typing import List, Tuple


class AnatomicalConstraintModule(nn.Module):
    """
    Anatomical Constraint Module (ACM).
    
    Projects raw network predictions onto a learned manifold of physiologically
    feasible eye rotations.
    """
    
    def __init__(
        self,
        yaw_max: float = 55.0,
        pitch_max: float = 40.0,
        hidden_dims: List[int] = [64, 32]
    ):
        super().__init__()
        
        self.yaw_max = yaw_max
        self.pitch_max = pitch_max
        
        layers = []
        in_dim = 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        
        self.mlp = nn.Sequential(*layers)
        
        self.register_buffer('yaw_limit', torch.tensor(yaw_max))
        self.register_buffer('pitch_limit', torch.tensor(pitch_max))
    
    def forward(self, gaze_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.mlp(gaze_init)
        gaze_valid = torch.tanh(gaze_init + residual)
        correction = gaze_valid - gaze_init
        
        return gaze_valid, correction
    
    def is_anatomically_valid(self, gaze: torch.Tensor) -> torch.Tensor:
        yaw_deg = gaze[:, 0] * self.yaw_max
        pitch_deg = gaze[:, 1] * self.pitch_max
        
        outlier_yaw = 60.0
        outlier_pitch = 45.0
        
        valid = (torch.abs(yaw_deg) <= outlier_yaw) & (torch.abs(pitch_deg) <= outlier_pitch)
        return valid