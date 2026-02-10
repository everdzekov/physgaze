import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicalConstraintModule(nn.Module):
    """
    Anatomical Constraint Module (ACM) from PhysGaze paper
    Projects gaze predictions onto physiologically plausible manifold
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        
        # MLP for learning the correction
        self.correction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # Physiological limits (from paper: ±55° yaw, ±40° pitch)
        self.yaw_limit = 50.0
        self.pitch_limit = 35.0
        
        # Learnable scale for corrections
        self.correction_scale = nn.Parameter(torch.tensor(0.1))
        
        # Initialize to small corrections
        nn.init.normal_(self.correction_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.correction_net[-1].bias)
    
    def forward(self, gaze_angles: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        """
        Apply anatomical constraints to gaze predictions
        
        Args:
            gaze_angles: [batch_size, 2] tensor of (yaw, pitch) in degrees
            mode: 'train' (learned correction) or 'inference' (hard clipping)
        """
        batch_size = gaze_angles.size(0)
        
        if mode == 'inference':
            # Hard clipping for inference (as in paper)
            yaw = torch.clamp(gaze_angles[:, 0], -self.yaw_limit, self.yaw_limit)
            pitch = torch.clamp(gaze_angles[:, 1], -self.pitch_limit, self.pitch_limit)
            return torch.stack([yaw, pitch], dim=1)
        
        # During training: learn smooth correction
        gaze_normalized = gaze_angles.clone()
        gaze_normalized[:, 0] = gaze_normalized[:, 0] / self.yaw_limit
        gaze_normalized[:, 1] = gaze_normalized[:, 1] / self.pitch_limit
        
        # Get correction from network
        correction = self.correction_net(gaze_normalized)
        correction = correction * self.correction_scale
        
        # Apply correction
        gaze_corrected = gaze_angles + correction
        
        # Apply soft constraints (allows gradient flow)
        yaw_corrected = self.yaw_limit * torch.tanh(gaze_corrected[:, 0] / self.yaw_limit)
        pitch_corrected = self.pitch_limit * torch.tanh(gaze_corrected[:, 1] / self.pitch_limit)
        
        return torch.stack([yaw_corrected, pitch_corrected], dim=1)
    
    def compute_outliers(self, gaze_angles: torch.Tensor) -> float:
        """Compute percentage of physiologically impossible predictions"""
        yaw_outliers = torch.abs(gaze_angles[:, 0]) > self.yaw_limit
        pitch_outliers = torch.abs(gaze_angles[:, 1]) > self.pitch_limit
        outliers = torch.logical_or(yaw_outliers, pitch_outliers)
        return outliers.float().mean().item()
    
    def regularization_loss(self, gaze_input: torch.Tensor, 
                          gaze_output: torch.Tensor) -> torch.Tensor:
        """Encourage minimal corrections"""
        return F.mse_loss(gaze_output, gaze_input) * 0.05
