import os
import json
import numpy as np
import torch
import random
from typing import Dict, Any, Optional
import yaml

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"💾 Saved checkpoint to: {path}")

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   path: str, device: torch.device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"📂 Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return model, optimizer, epoch, loss

def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 Saved config to: {path}")

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directory(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created directory: {path}")

def get_device() -> torch.device:
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using MPS device")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU device")
    
    return device
