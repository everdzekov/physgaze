import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.mpiigaze_dataset import MPIIGazeDataset
from models.gaze_networks import SimpleGazeNet

def evaluate_mpiigaze():
    """Evaluate on MPIIGaze test set."""
    # Load configuration
    with open('configs/mpiigaze_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    test_dataset = MPIIGazeDataset(
        split='test',
        test_subject=config['data']['test_subject'],
        samples_per_subject=config['data']['samples_per_subject'],
        augment=False,
        debug=False,
        filter_extreme_gaze=config['data']['filter_extreme_gaze']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    model = SimpleGazeNet().to(device)
    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'best_mpiigaze_model.pth')
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        test_mae = 0
        test_samples = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                
                # Denormalize
                outputs_deg = outputs * torch.tensor([30.0, 20.0]).to(device)
                targets_deg = targets * torch.tensor([30.0, 20.0]).to(device)
                
                mae = torch.mean(torch.abs(outputs_deg - targets_deg)).item()
                test_mae += mae * images.size(0)
                test_samples += images.size(0)
        
        if test_samples > 0:
            test_mae /= test_samples
            print(f"✅ Test MAE: {test_mae:.2f}°")
            return test_mae
        else:
            print("⚠️ No test samples available")
    else:
        print(f"❌ Model checkpoint not found at {checkpoint_path}")
    
    return None

if __name__ == "__main__":
    evaluate_mpiigaze()
