#!/usr/bin/env python3
"""
PhysGaze Training Script
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loaders import MPIIGazeLoader, Gaze360Loader
from data.datasets import GazeDataset
from models.physgaze import PhysGaze
from training.trainer import EnhancedPhysGazeTrainer
from utils.helpers import set_seed, get_device, create_directory
from configs import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train PhysGaze model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='MPIIGaze',
                       choices=['MPIIGaze', 'Gaze360'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data/MPIIGaze',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # Load config
    config = get_config(args.config)
    config.update(vars(args))
    
    # Create directories
    create_directory(args.save_dir)
    create_directory('./results')
    create_directory('./visualizations')
    
    # Load dataset
    print(f"\n📁 Loading {args.dataset} dataset...")
    
    if args.dataset == 'MPIIGaze':
        loader = MPIIGazeLoader(base_dir=args.data_dir)
        df = loader.load_all_subjects()
        
        # For training, we'll use all data (not LOSO)
        train_df = df.copy()
        
    elif args.dataset == 'Gaze360':
        loader = Gaze360Loader(base_dir=args.data_dir)
        train_df = loader.load_data(split='train')
    
    # Create dataset and dataloader
    train_dataset = GazeDataset(train_df, augment=True)
    val_dataset = GazeDataset(train_df, augment=False)
    
    # Split into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\n📊 Dataset statistics:")
    print(f"  Train samples: {len(train_subset):,}")
    print(f"  Validation samples: {len(val_subset):,}")
    
    # Create model
    print(f"\n🧠 Creating PhysGaze model...")
    model = PhysGaze(image_size=tuple(config['data']['image_size']))
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = EnhancedPhysGazeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = trainer.train(max_epochs=args.epochs)
    
    # Save final model
    checkpoint_path = os.path.join(args.save_dir, 'physgaze_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, checkpoint_path)
    
    print(f"\n🎉 Training completed!")
    print(f"  Final model saved to: {checkpoint_path}")
    print(f"  Best validation MAE: {trainer.best_val_mae:.2f}°")
    print(f"  Total training epochs: {len(history['train_loss'])}")

if __name__ == '__main__':
    main()
