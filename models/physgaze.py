import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import GazeEstimationBackbone
from .acm import AnatomicalConstraintModule
from .renderer import SimpleRenderer

class PhysGaze(nn.Module):
    """
    Complete PhysGaze model for gaze estimation
    """
    
    def __init__(self, image_size: tuple = (36, 60)):
        super().__init__()
        
        self.image_size = image_size
        
        # Backbone (feature extractor)
        self.backbone = GazeEstimationBackbone(input_channels=1)
        
        # Gaze regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Scale factors (MPIIGaze ranges: yaw ±45°, pitch ±30°)
        self.register_buffer('yaw_scale', torch.tensor(45.0))
        self.register_buffer('pitch_scale', torch.tensor(30.0))
        
        # Anatomical Constraint Module
        self.acm = AnatomicalConstraintModule()
        
        # Differentiable Renderer
        self.renderer = SimpleRenderer(image_size=image_size)
        
        # Loss weights (from paper: λ_reg=1.0, λ_cycle=0.2, λ_acm=0.1)
        self.lambda_reg = 1.0
        self.lambda_cycle = 0.2
        self.lambda_acm = 0.1
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Forward pass through PhysGaze
        
        Args:
            x: Input image tensor [batch, 1, H, W]
            return_all: Whether to return intermediate outputs
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Initial gaze prediction (normalized to [-1, 1])
        gaze_norm = self.regressor(features)
        
        # Convert to degrees
        yaw = gaze_norm[:, 0] * self.yaw_scale
        pitch = gaze_norm[:, 1] * self.pitch_scale
        gaze_init = torch.stack([yaw, pitch], dim=1)
        
        # Apply Anatomical Constraint Module
        gaze_corrected = self.acm(gaze_init, mode='train')
        
        # Render synthetic image
        rendered_img = self.renderer(gaze_corrected)
        
        if return_all:
            return {
                'gaze_init': gaze_init,
                'gaze_corrected': gaze_corrected,
                'rendered_img': rendered_img,
                'features': features
            }
        else:
            return gaze_corrected
    
    def compute_losses(self, predictions: dict, targets: torch.Tensor, 
                      input_images: torch.Tensor) -> dict:
        """
        Compute PhysGaze losses (Equation 6 in paper)
        """
        gaze_init = predictions['gaze_init']
        gaze_corrected = predictions['gaze_corrected']
        rendered_img = predictions['rendered_img']
        
        # 1. Regression loss (MAE)
        reg_loss = F.l1_loss(gaze_corrected, targets)
        
        # 2. Cycle-consistency loss (L1)
        cycle_loss = F.l1_loss(rendered_img, input_images)
        
        # 3. ACM regularization loss
        acm_loss = self.acm.regularization_loss(gaze_init, gaze_corrected)
        
        # Total loss (Equation 6)
        total_loss = (self.lambda_reg * reg_loss +
                     self.lambda_cycle * cycle_loss +
                     self.lambda_acm * acm_loss)
        
        return {
            'total': total_loss,
            'reg': reg_loss,
            'cycle': cycle_loss,
            'acm': acm_loss
        }
    
    def get_outlier_rate(self, gaze_predictions: torch.Tensor) -> float:
        """Get percentage of physiologically impossible predictions"""
        return self.acm.compute_outliers(gaze_predictions) * 100
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode with hard clipping"""
        features = self.backbone(x)
        gaze_norm = self.regressor(features)
        
        yaw = gaze_norm[:, 0] * self.yaw_scale
        pitch = gaze_norm[:, 1] * self.pitch_scale
        gaze_init = torch.stack([yaw, pitch], dim=1)
        
        # Apply ACM in inference mode
        gaze_corrected = self.acm(gaze_init, mode='inference')
        
        return gaze_corrected
