import torch
import torch.nn as nn
import torch.nn.functional as F

class GazeEstimationBackbone(nn.Module):
    """
    Feature extraction backbone (ResNet-18 inspired)
    Adapted for single-channel input
    """
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        # Simplified ResNet-like architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 36, 60)
            dummy = self.conv1(dummy)
            dummy = self.bn1(dummy)
            dummy = self.relu(dummy)
            dummy = self.maxpool(dummy)
            dummy = self.layer1(dummy)
            dummy = self.layer2(dummy)
            dummy = self.layer3(dummy)
            dummy = self.avgpool(dummy)
            self.out_features = dummy.numel()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        
        # First block
        layers.append(self._residual_block(in_channels, out_channels, stride))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels: int, out_channels: int, 
                       stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
