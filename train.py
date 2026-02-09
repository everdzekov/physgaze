import os
import sys
import json
import torch
from datetime import datetime
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.mpiigaze_dataset import MPIIGazeDataset
from models.gaze_networks import SimpleGazeNet
from training.trainer import GazeTrainer
from utils.visualization import plot_training_history
from utils.gaze_utils import debug_gaze_values
from torch.utils.data import DataLoader

def load_config(config_path='configs/mpiigaze_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main training function."""
    print("=" * 70)
    print("👁️  MPIIGAZE GAZE ESTIMATION TRAINING")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(42)
    
    # Debug gaze values
    print("\n1. Debugging gaze values...")
    debug_gaze_values(base_dir=config['data']['base_dir'])
    
    # Create datasets
    print("\n2. Creating datasets...")
    
    train_dataset = MPIIGazeDataset(
        split='train',
        test_subject=config['data']['test_subject'],
        samples_per_subject=config['data']['samples_per_subject'],
        augment=True,
        debug=False,
        filter_extreme_gaze=config['data']['filter_extreme_gaze']
    )
    
    val_dataset = MPIIGazeDataset(
        split='val',
        test_subject=config['data']['test_subject'],
        samples_per_subject=config['data']['samples_per_subject'],
        augment=False,
        debug=False,
        filter_extreme_gaze=config['data']['filter_extreme_gaze']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n3. Creating model...")
    model = SimpleGazeNet()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\n4. Creating trainer...")
    trainer = GazeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config['training']['learning_rate']
    )
    
    # Create output directories
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['outputs'], exist_ok=True)
    
    # Train model
    print("\n5. Starting training...")
    history = trainer.train(epochs=config['training']['epochs'])
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    torch.save(model.state_dict(), 
               os.path.join(config['paths']['checkpoints'], 'mpiigaze_final.pth'))
    
    # Save results
    results = {
        'best_val_mae': float(trainer.best_val_mae),
        'epochs': config['training']['epochs'],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_subject': config['data']['test_subject'],
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(config['paths']['outputs'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_path}")
    print("=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
