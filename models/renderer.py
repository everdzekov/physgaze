import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRenderer(nn.Module):
    """
    Simplified differentiable renderer for cycle-consistency
    (Paper uses PyTorch3D, but we use a simpler 2D warping approach)
    """
    
    def __init__(self, image_size: tuple = (36, 60)):
        super().__init__()
        
        self.image_size = image_size
        self.H, self.W = image_size
        
        # Base eye template (learnable)
        self.base_template = nn.Parameter(
            torch.randn(1, 1, self.H, self.W) * 0.1
        )
        
        # Gaze-dependent warping network
        self.warp_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.H * self.W * 2)  # Output flow field
        )
        
        # Initialize to identity warp
        nn.init.zeros_(self.warp_net[-1].weight)
        nn.init.zeros_(self.warp_net[-1].bias)
    
    def forward(self, gaze_angles: torch.Tensor) -> torch.Tensor:
        batch_size = gaze_angles.size(0)
        
        # Normalize gaze
        gaze_norm = gaze_angles.clone()
        gaze_norm[:, 0] = gaze_norm[:, 0] / 45.0
        gaze_norm[:, 1] = gaze_norm[:, 1] / 30.0
        
        # Generate flow field
        flow_params = self.warp_net(gaze_norm)
        flow = flow_params.view(batch_size, 2, self.H, self.W)
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.H, device=gaze_angles.device),
            torch.linspace(-1, 1, self.W, device=gaze_angles.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid = grid.repeat(batch_size, 1, 1, 1)
        
        # Apply flow
        flow_norm = flow.permute(0, 2, 3, 1) / torch.tensor([self.W/2, self.H/2],
                                                          device=gaze_angles.device)
        warped_grid = grid + flow_norm
        
        # Warp base template
        base = self.base_template.expand(batch_size, -1, -1, -1)
        rendered = F.grid_sample(base, warped_grid, mode='bilinear', align_corners=True)
        
        return rendered
