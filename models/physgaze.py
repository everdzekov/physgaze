import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from models.backbone import ResNetBackbone
from models.acm import AnatomicalConstraintModule
from models.renderer import DifferentiableRenderer


class PhysGaze(nn.Module):
    """
    PhysGaze: Physics-Informed Deep Learning Framework for Gaze Estimation.
    """
    
    def __init__(
        self,
        pretrained_backbone: bool = True,
        use_acm: bool = True,
        use_renderer: bool = True,
        image_size: Tuple[int, int] = (36, 60)
    ):
        super().__init__()
        
        self.use_acm = use_acm
        self.use_renderer = use_renderer
        self.image_size = image_size
        
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        
        if use_acm:
            self.acm = AnatomicalConstraintModule()
        
        if use_renderer:
            self.renderer = DifferentiableRenderer(image_size=image_size)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        if x.shape[2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        
        gaze_init = self.backbone(x)
        
        if self.use_acm:
            gaze, correction = self.acm(gaze_init)
        else:
            gaze = gaze_init
            correction = torch.zeros_like(gaze_init)
        
        outputs = {
            'gaze': gaze,
            'gaze_init': gaze_init,
            'correction': correction
        }
        
        if self.use_renderer and return_all:
            rendered = self.renderer(gaze)
            outputs['rendered'] = rendered
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x, return_all=False)
        return outputs['gaze']