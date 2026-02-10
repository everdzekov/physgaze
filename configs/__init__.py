import os
import yaml
from typing import Dict, Any

def get_config(config_path: str = 'configs/default.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        # Return default config if file doesn't exist
        return {
            'data': {
                'dataset': 'MPIIGaze',
                'data_dir': './data/MPIIGaze',
                'image_size': [36, 60],
                'batch_size': 128,
                'augment': True,
                'validation_split': 0.2
            },
            'model': {
                'input_channels': 1,
                'hidden_dim': 64,
                'dropout': 0.3,
                'lambda_reg': 1.0,
                'lambda_cycle': 0.2,
                'lambda_acm': 0.1,
                'yaw_limit': 50.0,
                'pitch_limit': 35.0
            },
            'training': {
                'max_epochs': 100,
                'lr': 1e-4,
                'weight_decay': 1e-4,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.001,
                'overfitting_threshold': 0.1,
                'scheduler_patience': 5,
                'gradient_clip': 1.0
            },
            'evaluation': {
                'thresholds': [5.0, 10.0, 15.0],
                'save_results': True,
                'save_visualizations': True
            },
            'paths': {
                'checkpoints': './checkpoints',
                'results': './results',
                'visualizations': './visualizations'
            }
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
