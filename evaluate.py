#!/usr/bin/env python3
"""
PhysGaze Evaluation Script
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loaders import MPIIGazeLoader
from models.physgaze import PhysGaze
from evaluation.evaluator import EnhancedLOSOEvaluator
from utils.helpers import set_seed, get_device, create_directory

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PhysGaze model with LOSO')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/MPIIGaze',
                       help='Path to MPIIGaze dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum epochs per fold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
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
    
    # Load dataset
    print(f"\n📁 Loading MPIIGaze dataset...")
    loader = MPIIGazeLoader(base_dir=args.data_dir)
    df = loader.load_all_subjects()
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,
        'overfitting_threshold': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'scheduler_patience': 3,
        'image_size': (36, 60)
    }
    
    # Create directories
    create_directory('./results')
    create_directory('./visualizations')
    
    # Run LOSO evaluation
    print(f"\n🎯 Starting LOSO evaluation...")
    evaluator = EnhancedLOSOEvaluator(df, config)
    results = evaluator.run(device)
    
    print(f"\n🎉 LOSO evaluation completed!")
    
    # Print summary
    if hasattr(evaluator, 'fold_histories'):
        total_epochs = sum([len(h['train_loss']) for h in evaluator.fold_histories.values()])
        original_epochs = len(evaluator.all_subjects) * args.max_epochs
        
        print(f"\n⚡ Training Efficiency:")
        print(f"  Original (fixed epochs): {original_epochs} total epochs")
        print(f"  Actual (dynamic stopping): {total_epochs} total epochs")
        print(f"  Epochs saved: {original_epochs - total_epochs} "
              f"({(original_epochs - total_epochs)/original_epochs*100:.1f}% reduction)")
        print(f"  Average epochs per fold: {total_epochs/len(evaluator.fold_histories):.1f}")

if __name__ == '__main__':
    main()
