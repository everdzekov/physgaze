#!/usr/bin/env python3
"""
Evaluation script for PhysGaze.
"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from data.mpiigaze import MPIIGazeDataset
from data.preprocessing import SyntheticMPIIGaze
from models.physgaze import PhysGaze
from evaluation.evaluator import PhysGazeEvaluator
from utils.visualization import plot_results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PhysGaze model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['mpii', 'synthetic'], default='synthetic', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data/MPIIGaze', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./outputs/results', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = PhysGaze(
        pretrained_backbone=False,
        use_acm=True,
        use_renderer=True,
        image_size=(36, 60)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset
    if args.dataset == 'mpii':
        test_dataset = MPIIGazeDataset(root_dir=args.data_dir, split='test')
    else:
        test_dataset = SyntheticMPIIGaze(num_samples=1000, split='test')
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    evaluator = PhysGazeEvaluator(model, device)
    metrics, preds_deg, targets_deg, angular_errors = evaluator.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS:")
    print("=" * 60)
    print(f"Mean Angular Error: {metrics['mae']:.2f}°")
    print(f"Yaw MAE: {metrics['mae_yaw']:.2f}°")
    print(f"Pitch MAE: {metrics['mae_pitch']:.2f}°")
    print(f"Error Std: {metrics['std']:.2f}°")
    print(f"Median Error: {metrics['median']:.2f}°")
    print(f"95th Percentile: {metrics['p95']:.2f}°")
    print(f"Outlier Rate: {metrics['outlier_rate']:.1f}%")
    print(f"Mean Correction: {metrics['mean_correction']:.4f}")
    print("=" * 60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(
        preds_deg, targets_deg, angular_errors,
        save_path=str(output_dir / 'results.png')
    )
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    main()