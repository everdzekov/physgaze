import torch
import torch.nn as nn
from typing import Tuple


class DifferentiableRenderer(nn.Module):
    """
    Differentiable Renderer for cycle-consistency.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (36, 60),
        texture_size: int = 64
    ):
        super().__init__()
        
        self.image_size = image_size
        self.texture_size = texture_size
        
        self.texture = nn.Parameter(
            torch.randn(1, texture_size, texture_size) * 0.1 + 0.5
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, image_size[0] * image_size[1]),
            nn.Sigmoid()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, gaze: torch.Tensor) -> torch.Tensor:
        batch_size = gaze.shape[0]
        h, w = self.image_size
        
        x = self.decoder(gaze)
        x = x.view(batch_size, 1, h, w)
        rendered = self.refine(x)
        
        return rendered