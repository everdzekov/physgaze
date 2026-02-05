#!/usr/bin/env python3
"""
Main entry point for PhysGaze.
Combines training, evaluation, and inference functionalities.
"""

import argparse
import torch
from pathlib import Path

from utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser(description='PhysGaze: Physics-Informed Gaze Estimation')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], default='train', help='Mode to run')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        from train import main as train_main
        train_main()
    elif args.mode == 'evaluate':
        from evaluate import main as eval_main
        eval_main()
    elif args.mode == 'inference':
        from inference import main as infer_main
        infer_main()
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()