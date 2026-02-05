import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysGazeLoss(nn.Module):
    """
    Combined loss function for PhysGaze.
    """
    
    def __init__(
        self,
        lambda_reg: float = 1.0,
        lambda_cycle: float = 0.2,
        lambda_acm: float = 0.1,
        yaw_max: float = 55.0,
        pitch_max: float = 40.0
    ):
        super().__init__()
        
        self.lambda_reg = lambda_reg
        self.lambda_cycle = lambda_cycle
        self.lambda_acm = lambda_acm
        self.yaw_max = yaw_max
        self.pitch_max = pitch_max
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        gaze = outputs['gaze']
        gaze_init = outputs.get('gaze_init', gaze)
        correction = outputs.get('correction', torch.zeros_like(gaze))
        rendered = outputs.get('rendered', None)
        
        loss_reg = F.l1_loss(gaze, targets)
        
        if rendered is not None and inputs is not None:
            if inputs.shape != rendered.shape:
                inputs_resized = F.interpolate(
                    inputs, size=rendered.shape[2:],
                    mode='bilinear', align_corners=False
                )
            else:
                inputs_resized = inputs
            loss_cycle = F.l1_loss(rendered, inputs_resized)
        else:
            loss_cycle = torch.tensor(0.0, device=gaze.device)
        
        loss_acm = torch.mean(correction ** 2)
        
        loss_total = (
            self.lambda_reg * loss_reg +
            self.lambda_cycle * loss_cycle +
            self.lambda_acm * loss_acm
        )
        
        with torch.no_grad():
            gaze_deg = gaze.clone()
            gaze_deg[:, 0] *= self.yaw_max
            gaze_deg[:, 1] *= self.pitch_max
            
            targets_deg = targets.clone()
            targets_deg[:, 0] *= self.yaw_max
            targets_deg[:, 1] *= self.pitch_max
            
            angular_error = torch.mean(torch.abs(gaze_deg - targets_deg))
        
        return {
            'total': loss_total,
            'reg': loss_reg,
            'cycle': loss_cycle,
            'acm': loss_acm,
            'angular_error': angular_error
        }