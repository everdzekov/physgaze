#!/usr/bin/env python3
"""
Main training script for PhysGaze.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path

from data.mpiigaze import MPIIGazeDataset, download_mpiigaze
from data.preprocessing import SyntheticMPIIGaze
from models.physgaze import PhysGaze
from training.trainer import PhysGazeTrainer
from utils.helpers import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train PhysGaze model')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['mpii', 'synthetic'], default='synthetic', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data/MPIIGaze', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs/physgaze', help='Log directory')
    parser.add_argument('--use_real_data', action='store_true', help='Use real MPIIGaze data')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    if args.use_real_data or args.dataset == 'mpii':
        print("Loading MPIIGaze dataset...")
        
        if not Path(args.data_dir).exists():
            print("Downloading dataset...")
            download_mpiigaze(args.data_dir)
        
        train_dataset = MPIIGazeDataset(root_dir=args.data_dir, split='train')
        val_dataset = MPIIGazeDataset(root_dir=args.data_dir, split='val')
        test_dataset = MPIIGazeDataset(root_dir=args.data_dir, split='test')
    else:
        print("Using synthetic dataset...")
        train_dataset = SyntheticMPIIGaze(num_samples=8000, split='train')
        val_dataset = SyntheticMPIIGaze(num_samples=1000, split='val')
        test_dataset = SyntheticMPIIGaze(num_samples=1000, split='test')
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = PhysGaze(
        pretrained_backbone=True,
        use_acm=True,
        use_renderer=True,
        image_size=(36, 60)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = PhysGazeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        log_dir=args.log_dir
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    print("\nTraining completed!")
    print(f"Best validation MAE: {trainer.best_val_mae:.2f}°")


if __name__ == '__main__':
    main()