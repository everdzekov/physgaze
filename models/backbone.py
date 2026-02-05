import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNetBackbone(nn.Module):
    """
    ResNet-18 based backbone for feature extraction.
    Adapted for single-channel eye images.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        resnet = resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)
                )
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        gaze = self.fc(x)
        return gaze