"""
PhysGaze with LOSO (Leave-One-Subject-Out) evaluation on MPIIGaze
Following the exact protocol from the PhysGaze paper
WITH DYNAMIC EPOCH STOPPING AND ENHANCED VISUALIZATIONS
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import json
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PROPER MPIIGAZE DATA LOADER WITH LOSO SUPPORT
# ============================================================================

class MPIIGazeLoader:
    """
    Loads MPIIGaze dataset with proper LOSO splits
    Following the exact protocol from the PhysGaze paper
    """

    @staticmethod
    def load_all_subjects(base_dir='./data/MPIIGaze/MPIIGaze'):
        """
        Load all MPIIGaze subjects following paper's protocol
        Paper uses 15 subjects: p00 to p14
        """
        print(f"Loading MPIIGaze dataset from: {base_dir}")

        # Standard MPIIGaze structure
        normalized_dir = os.path.join(base_dir, "Data", "Normalized")

        # Check alternative paths
        if not os.path.exists(normalized_dir):
            alt_paths = [
                './MPIIGaze/Data/Normalized',
                './data/MPIIGaze/Normalized',
                normalized_dir
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    normalized_dir = path
                    print(f"Found data at: {path}")
                    break
            else:
                raise FileNotFoundError(f"MPIIGaze data not found. Checked: {alt_paths}")

        # Get all subjects (p00 to p14)
        subjects = sorted([d for d in os.listdir(normalized_dir)
                          if os.path.isdir(os.path.join(normalized_dir, d)) and d.startswith('p')])

        print(f"Found {len(subjects)} subjects: {subjects}")

        all_data = []

        for subject in tqdm(subjects, desc="Loading subjects"):
            subject_path = os.path.join(normalized_dir, subject)
            mat_files = sorted([f for f in os.listdir(subject_path)
                              if f.endswith('.mat') and f.startswith('day')])

            for mat_file in mat_files:
                mat_path = os.path.join(subject_path, mat_file)

                try:
                    mat_data = sio.loadmat(mat_path, simplify_cells=True)

                    if 'data' not in mat_data:
                        continue

                    data_dict = mat_data['data']

                    # Process right eye
                    if 'right' in data_dict:
                        right_data = data_dict['right']
                        if isinstance(right_data, dict):
                            if 'image' in right_data and 'gaze' in right_data:
                                images = right_data['image']
                                gazes = right_data['gaze']

                                # Convert to proper format
                                if images.ndim == 4:  # (n, 36, 60, 3)
                                    images = np.mean(images, axis=3)  # Convert to grayscale

                                n_samples = min(len(images), len(gazes))

                                for i in range(n_samples):
                                    # Process image
                                    img = images[i].astype(np.float32) / 255.0

                                    # Process gaze (convert 3D vector to angles)
                                    gaze_3d = gazes[i]
                                    yaw, pitch = MPIIGazeLoader._gaze_vector_to_angles(gaze_3d)

                                    all_data.append({
                                        'subject': subject,
                                        'image': img,
                                        'gaze_3d': gaze_3d,
                                        'gaze_yaw': yaw,
                                        'gaze_pitch': pitch,
                                        'eye': 'right'
                                    })

                    # Process left eye
                    if 'left' in data_dict:
                        left_data = data_dict['left']
                        if isinstance(left_data, dict):
                            if 'image' in left_data and 'gaze' in left_data:
                                images = left_data['image']
                                gazes = left_data['gaze']

                                # Convert to proper format
                                if images.ndim == 4:  # (n, 36, 60, 3)
                                    images = np.mean(images, axis=3)  # Convert to grayscale

                                n_samples = min(len(images), len(gazes))

                                for i in range(n_samples):
                                    # Process image
                                    img = images[i].astype(np.float32) / 255.0

                                    # Process gaze (convert 3D vector to angles)
                                    gaze_3d = gazes[i]
                                    yaw, pitch = MPIIGazeLoader._gaze_vector_to_angles(gaze_3d)

                                    all_data.append({
                                        'subject': subject,
                                        'image': img,
                                        'gaze_3d': gaze_3d,
                                        'gaze_yaw': yaw,
                                        'gaze_pitch': pitch,
                                        'eye': 'left'
                                    })

                except Exception as e:
                    print(f"Error loading {mat_file}: {e}")
                    continue

        df = pd.DataFrame(all_data)

        if len(df) == 0:
            print("No data loaded. Creating synthetic data for testing...")
            return MPIIGazeLoader._create_synthetic_data()

        print(f"\n✅ Dataset loaded successfully!")
        print(f"Total samples: {len(df):,}")
        print(f"Subjects: {df['subject'].unique().tolist()}")
        print(f"Eye distribution: {df['eye'].value_counts().to_dict()}")

        return df

    @staticmethod
    def _gaze_vector_to_angles(gaze_vector):
        """
        Convert 3D gaze vector to yaw and pitch angles (in degrees)
        Using MPIIGaze formula: yaw = atan2(-x, -z), pitch = arcsin(-y)
        """
        if gaze_vector is None or len(gaze_vector) < 3:
            return 0.0, 0.0

        x, y, z = gaze_vector[0], gaze_vector[1], gaze_vector[2]

        # Normalize
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 0:
            x, y, z = x/norm, y/norm, z/norm

        # Convert to angles (following MPIIGaze convention)
        yaw = np.arctan2(-x, -z)  # Horizontal angle
        pitch = np.arcsin(-y)      # Vertical angle

        # Convert to degrees
        yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)

        # Clip to reasonable ranges
        yaw_deg = np.clip(yaw_deg, -45, 45)
        pitch_deg = np.clip(pitch_deg, -30, 30)

        return yaw_deg, pitch_deg

    @staticmethod
    def _create_synthetic_data():
        """Create synthetic MPIIGaze-like data for testing"""
        print("Creating synthetic MPIIGaze data...")

        subjects = [f'p{i:02d}' for i in range(15)]  # p00 to p14
        all_data = []

        for subject in subjects:
            n_samples = np.random.randint(8000, 15000)  # Varying samples per subject

            for i in range(n_samples):
                # Create synthetic eye image
                img = np.random.rand(36, 60).astype(np.float32) * 0.3

                # Add eye-like structure
                center_y, center_x = 18, 30
                for y in range(36):
                    for x in range(60):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < 8:
                            img[y, x] = 0.1  # Pupil
                        elif dist < 15:
                            img[y, x] = 0.5  # Iris

                # Generate gaze angles (mostly within limits, some extremes)
                if np.random.random() < 0.15:  # 15% outliers
                    yaw = np.random.uniform(-70, 70)
                    pitch = np.random.uniform(-50, 50)
                else:
                    yaw = np.random.uniform(-40, 40)
                    pitch = np.random.uniform(-25, 25)

                # Add subject-specific bias
                subject_id = int(subject[1:])
                yaw += subject_id * 2.0  # Each subject has different bias

                all_data.append({
                    'subject': subject,
                    'image': img,
                    'gaze_3d': [np.sin(np.radians(yaw)), np.sin(np.radians(pitch)), 1.0],
                    'gaze_yaw': yaw,
                    'gaze_pitch': pitch,
                    'eye': 'right' if np.random.random() < 0.5 else 'left'
                })

        df = pd.DataFrame(all_data)
        print(f"Created synthetic data: {len(df):,} samples from {len(subjects)} subjects")
        return df

# ============================================================================
# PHYSGAZE MODEL COMPONENTS
# ============================================================================

class AnatomicalConstraintModule(nn.Module):
    """
    Anatomical Constraint Module (ACM) from PhysGaze paper
    Projects gaze predictions onto physiologically plausible manifold
    """
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()

        # MLP for learning the correction
        self.correction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

        # Physiological limits (from paper: ±55° yaw, ±40° pitch)
        # Using ±50° yaw and ±35° pitch for safety margin
        self.yaw_limit = 50.0
        self.pitch_limit = 35.0

        # Learnable scale for corrections
        self.correction_scale = nn.Parameter(torch.tensor(0.1))

        # Initialize to small corrections
        nn.init.normal_(self.correction_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.correction_net[-1].bias)

    def forward(self, gaze_angles, mode='train'):
        """
        Apply anatomical constraints to gaze predictions

        Args:
            gaze_angles: [batch_size, 2] tensor of (yaw, pitch) in degrees
            mode: 'train' (learned correction) or 'inference' (hard clipping)
        """
        batch_size = gaze_angles.size(0)

        if mode == 'inference':
            # Hard clipping for inference (as in paper)
            yaw = torch.clamp(gaze_angles[:, 0], -self.yaw_limit, self.yaw_limit)
            pitch = torch.clamp(gaze_angles[:, 1], -self.pitch_limit, self.pitch_limit)
            return torch.stack([yaw, pitch], dim=1)

        # During training: learn smooth correction
        # Normalize for better learning
        gaze_normalized = gaze_angles.clone()
        gaze_normalized[:, 0] = gaze_normalized[:, 0] / self.yaw_limit
        gaze_normalized[:, 1] = gaze_normalized[:, 1] / self.pitch_limit

        # Get correction from network
        correction = self.correction_net(gaze_normalized)
        correction = correction * self.correction_scale

        # Apply correction
        gaze_corrected = gaze_angles + correction

        # Apply soft constraints (allows gradient flow)
        yaw_corrected = self.yaw_limit * torch.tanh(gaze_corrected[:, 0] / self.yaw_limit)
        pitch_corrected = self.pitch_limit * torch.tanh(gaze_corrected[:, 1] / self.pitch_limit)

        return torch.stack([yaw_corrected, pitch_corrected], dim=1)

    def compute_outliers(self, gaze_angles):
        """Compute percentage of physiologically impossible predictions"""
        yaw_outliers = torch.abs(gaze_angles[:, 0]) > self.yaw_limit
        pitch_outliers = torch.abs(gaze_angles[:, 1]) > self.pitch_limit
        outliers = torch.logical_or(yaw_outliers, pitch_outliers)
        return outliers.float().mean().item()

    def regularization_loss(self, gaze_input, gaze_output):
        """Encourage minimal corrections"""
        return F.mse_loss(gaze_output, gaze_input) * 0.05

class SimpleRenderer(nn.Module):
    """
    Simplified differentiable renderer for cycle-consistency
    (Paper uses PyTorch3D, but we use a simpler 2D warping approach)
    """
    def __init__(self, image_size=(36, 60)):
        super().__init__()

        self.image_size = image_size
        self.H, self.W = image_size

        # Base eye template (learnable)
        self.base_template = nn.Parameter(
            torch.randn(1, 1, self.H, self.W) * 0.1
        )

        # Gaze-dependent warping network
        self.warp_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.H * self.W * 2)  # Output flow field
        )

        # Initialize to identity warp
        nn.init.zeros_(self.warp_net[-1].weight)
        nn.init.zeros_(self.warp_net[-1].bias)

    def forward(self, gaze_angles):
        batch_size = gaze_angles.size(0)

        # Normalize gaze
        gaze_norm = gaze_angles.clone()
        gaze_norm[:, 0] = gaze_norm[:, 0] / 45.0
        gaze_norm[:, 1] = gaze_norm[:, 1] / 30.0

        # Generate flow field
        flow_params = self.warp_net(gaze_norm)
        flow = flow_params.view(batch_size, 2, self.H, self.W)

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.H, device=gaze_angles.device),
            torch.linspace(-1, 1, self.W, device=gaze_angles.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid = grid.repeat(batch_size, 1, 1, 1)

        # Apply flow
        flow_norm = flow.permute(0, 2, 3, 1) / torch.tensor([self.W/2, self.H/2],
                                                          device=gaze_angles.device)
        warped_grid = grid + flow_norm

        # Warp base template
        base = self.base_template.expand(batch_size, -1, -1, -1)
        rendered = F.grid_sample(base, warped_grid, mode='bilinear', align_corners=True)

        return rendered

class GazeEstimationBackbone(nn.Module):
    """
    Feature extraction backbone (ResNet-18 inspired)
    Paper uses ResNet-18 adapted for single-channel input
    """
    def __init__(self, input_channels=1):
        super().__init__()

        # Simplified ResNet-like architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 36, 60)
            dummy = self.conv1(dummy)
            dummy = self.bn1(dummy)
            dummy = self.relu(dummy)
            dummy = self.maxpool(dummy)
            dummy = self.layer1(dummy)
            dummy = self.layer2(dummy)
            dummy = self.layer3(dummy)
            dummy = self.avgpool(dummy)
            self.out_features = dummy.numel()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # First block
        layers.append(self._residual_block(in_channels, out_channels, stride))

        # Additional blocks
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def _residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class PhysGaze(nn.Module):
    """
    Complete PhysGaze model for LOSO evaluation
    """
    def __init__(self, image_size=(36, 60)):
        super().__init__()

        self.image_size = image_size

        # Backbone (feature extractor)
        self.backbone = GazeEstimationBackbone(input_channels=1)

        # Gaze regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Scale factors (MPIIGaze ranges: yaw ±45°, pitch ±30°)
        self.register_buffer('yaw_scale', torch.tensor(45.0))
        self.register_buffer('pitch_scale', torch.tensor(30.0))

        # Anatomical Constraint Module
        self.acm = AnatomicalConstraintModule()

        # Differentiable Renderer
        self.renderer = SimpleRenderer(image_size=image_size)

        # Loss weights (from paper: λ_reg=1.0, λ_cycle=0.2, λ_acm=0.1)
        self.lambda_reg = 1.0
        self.lambda_cycle = 0.2
        self.lambda_acm = 0.1

    def forward(self, x, return_all=False):
        """
        Forward pass through PhysGaze
        """
        # Feature extraction
        features = self.backbone(x)

        # Initial gaze prediction (normalized to [-1, 1])
        gaze_norm = self.regressor(features)

        # Convert to degrees
        yaw = gaze_norm[:, 0] * self.yaw_scale
        pitch = gaze_norm[:, 1] * self.pitch_scale
        gaze_init = torch.stack([yaw, pitch], dim=1)

        # Apply Anatomical Constraint Module
        gaze_corrected = self.acm(gaze_init, mode='train')

        # Render synthetic image
        rendered_img = self.renderer(gaze_corrected)

        if return_all:
            return {
                'gaze_init': gaze_init,
                'gaze_corrected': gaze_corrected,
                'rendered_img': rendered_img,
                'features': features
            }
        else:
            return gaze_corrected

    def compute_losses(self, predictions, targets, input_images):
        """
        Compute PhysGaze losses (Equation 6 in paper)
        """
        gaze_init = predictions['gaze_init']
        gaze_corrected = predictions['gaze_corrected']
        rendered_img = predictions['rendered_img']

        # 1. Regression loss (MAE)
        reg_loss = F.l1_loss(gaze_corrected, targets)

        # 2. Cycle-consistency loss (L1)
        cycle_loss = F.l1_loss(rendered_img, input_images)

        # 3. ACM regularization loss
        acm_loss = self.acm.regularization_loss(gaze_init, gaze_corrected)

        # Total loss (Equation 6)
        total_loss = (self.lambda_reg * reg_loss +
                     self.lambda_cycle * cycle_loss +
                     self.lambda_acm * acm_loss)

        return {
            'total': total_loss,
            'reg': reg_loss,
            'cycle': cycle_loss,
            'acm': acm_loss
        }

    def get_outlier_rate(self, gaze_predictions):
        """Get percentage of physiologically impossible predictions"""
        return self.acm.compute_outliers(gaze_predictions) * 100

# ============================================================================
# DYNAMIC EPOCH MANAGER WITH OVERFITTING DETECTION
# ============================================================================

class DynamicEpochManager:
    """
    Manages dynamic epoch stopping based on overfitting detection
    """
    def __init__(self, config):
        self.config = config
        self.patience = config.get('early_stopping_patience', 10)
        self.min_delta = config.get('early_stopping_min_delta', 0.001)
        self.overfitting_threshold = config.get('overfitting_threshold', 0.1)
        
        # Track history
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        
    def update(self, train_loss, val_loss, train_mae, val_mae):
        """Update metrics and check for overfitting"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_maes.append(train_mae)
        self.val_maes.append(val_mae)
        
        # Check for improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Check for overfitting (val loss increasing while train loss decreasing)
        if len(self.train_losses) > 5:
            recent_train_trend = self._calculate_trend(self.train_losses[-5:])
            recent_val_trend = self._calculate_trend(self.val_losses[-5:])
            
            # Overfitting detected when train loss is decreasing but val loss is increasing
            overfitting_detected = (recent_train_trend < -0.01 and recent_val_trend > 0.01 and 
                                   abs(recent_val_trend - recent_train_trend) > self.overfitting_threshold)
            
            if overfitting_detected:
                print(f"⚠️  Overfitting detected! Train trend: {recent_train_trend:.4f}, Val trend: {recent_val_trend:.4f}")
        
        # Check early stopping
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            print(f"🛑 Early stopping triggered after {len(self.train_losses)} epochs")
        
        return self.should_stop
    
    def _calculate_trend(self, values):
        """Calculate linear trend of values"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeff = np.polyfit(x, values, 1)
        return coeff[0]  # Slope
    
    def get_training_summary(self):
        """Get summary of training dynamics"""
        if len(self.train_losses) < 2:
            return None
            
        return {
            'epochs': len(self.train_losses),
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_mae': self.train_maes[-1],
            'final_val_mae': self.val_maes[-1],
            'best_val_loss': self.best_val_loss,
            'overfitting_ratio': (self.val_losses[-1] - self.train_losses[-1]) / self.train_losses[-1] if self.train_losses[-1] > 0 else 0,
            'train_val_gap': self.val_losses[-1] - self.train_losses[-1]
        }

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

class EnhancedVisualizer:
    """Enhanced visualization tools for training dynamics"""
    
    @staticmethod
    def plot_fold_training_curves(history, fold_name, save_path=None):
        """Plot comprehensive training curves for a single fold"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Training Dynamics - Fold: {fold_name}', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. Loss Curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend lines
        if len(epochs) > 2:
            z_train = np.polyfit(epochs, history['train_loss'], 1)
            z_val = np.polyfit(epochs, history['val_loss'], 1)
            p_train = np.poly1d(z_train)
            p_val = np.poly1d(z_val)
            axes[0, 0].plot(epochs, p_train(epochs), 'b--', alpha=0.5, linewidth=1)
            axes[0, 0].plot(epochs, p_val(epochs), 'r--', alpha=0.5, linewidth=1)
        
        # 2. MAE Curves
        axes[0, 1].plot(epochs, history['train_mae'], 'g-', label='Train MAE', linewidth=2)
        axes[0, 1].plot(epochs, history['val_mae'], 'm-', label='Val MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (°)')
        axes[0, 1].set_title('MAE Progression')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Outlier Rate
        axes[0, 2].plot(epochs, history['train_outliers'], 'c-', label='Train Outliers', linewidth=2)
        axes[0, 2].plot(epochs, history['val_outliers'], 'y-', label='Val Outliers', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Outlier Rate (%)')
        axes[0, 2].set_title('Outlier Reduction')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Loss Gap (Overfitting indicator)
        if len(history['train_loss']) > 0:
            loss_gap = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
            axes[1, 0].plot(epochs, loss_gap, 'k-', linewidth=2)
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 0].fill_between(epochs, 0, loss_gap, where=np.array(loss_gap) > 0, 
                                   color='red', alpha=0.3, label='Overfitting')
            axes[1, 0].fill_between(epochs, 0, loss_gap, where=np.array(loss_gap) < 0, 
                                   color='green', alpha=0.3, label='Underfitting')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Val Loss - Train Loss')
            axes[1, 0].set_title('Overfitting Indicator')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Learning Rate
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        # 6. Loss Ratio (Val/Train)
        if len(history['train_loss']) > 0:
            loss_ratio = [v/t if t > 0 else 0 for v, t in zip(history['val_loss'], history['train_loss'])]
            axes[1, 2].plot(epochs, loss_ratio, 'brown', linewidth=2)
            axes[1, 2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal Ratio')
            axes[1, 2].axhline(y=1.1, color='orange', linestyle='--', alpha=0.5, label='10% Overfit')
            axes[1, 2].axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='20% Overfit')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Val Loss / Train Loss')
            axes[1, 2].set_title('Overfitting Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Moving Average of MAE
        if len(history['val_mae']) > 5:
            window = min(5, len(history['val_mae']))
            val_mae_ma = pd.Series(history['val_mae']).rolling(window=window, center=True).mean()
            train_mae_ma = pd.Series(history['train_mae']).rolling(window=window, center=True).mean()
            
            axes[2, 0].plot(epochs, history['val_mae'], 'm-', alpha=0.3, label='Val MAE Raw')
            axes[2, 0].plot(epochs, val_mae_ma, 'm-', linewidth=2, label='Val MAE (MA)')
            axes[2, 0].plot(epochs, history['train_mae'], 'g-', alpha=0.3, label='Train MAE Raw')
            axes[2, 0].plot(epochs, train_mae_ma, 'g-', linewidth=2, label='Train MAE (MA)')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('MAE (°)')
            axes[2, 0].set_title('Moving Average of MAE')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Improvement Rate
        if len(history['val_mae']) > 2:
            improvement_rate = [-100 * (history['val_mae'][i] - history['val_mae'][i-1]) / 
                               history['val_mae'][i-1] if history['val_mae'][i-1] > 0 else 0 
                               for i in range(1, len(history['val_mae']))]
            axes[2, 1].plot(range(2, len(history['val_mae']) + 1), improvement_rate, 'orange', linewidth=2)
            axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Improvement Rate (%)')
            axes[2, 1].set_title('Epoch-to-Epoch Improvement')
            axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Training Summary Stats
        axes[2, 2].axis('off')
        if len(history['train_loss']) > 0:
            summary_text = (
                f"Training Summary:\n"
                f"Total Epochs: {len(epochs)}\n"
                f"Final Train MAE: {history['train_mae'][-1]:.2f}°\n"
                f"Final Val MAE: {history['val_mae'][-1]:.2f}°\n"
                f"Best Val MAE: {min(history['val_mae']):.2f}°\n"
                f"Overfitting Gap: {history['val_loss'][-1] - history['train_loss'][-1]:.4f}\n"
                f"Final Outlier Rate: {history['val_outliers'][-1]:.1f}%"
            )
            axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved training curves to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_gaze_vector_comparison(predictions, targets, initial_preds=None, title="Gaze Vector Comparison"):
        """Visualize gaze vectors in 2D space"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if initial_preds is not None and isinstance(initial_preds, torch.Tensor):
            initial_preds = initial_preds.cpu().numpy()
        
        # 1. Yaw vs Pitch scatter
        axes[0].scatter(targets[:, 0], targets[:, 1], alpha=0.5, label='Ground Truth', s=10)
        axes[0].scatter(predictions[:, 0], predictions[:, 1], alpha=0.5, label='Predictions', s=10)
        if initial_preds is not None:
            axes[0].scatter(initial_preds[:, 0], initial_preds[:, 1], alpha=0.3, label='Initial Preds', s=5)
        
        axes[0].set_xlabel('Yaw (degrees)')
        axes[0].set_ylabel('Pitch (degrees)')
        axes[0].set_title('Gaze Angle Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add physiological limits
        yaw_limit = 50
        pitch_limit = 35
        axes[0].axvline(x=-yaw_limit, color='r', linestyle='--', alpha=0.3)
        axes[0].axvline(x=yaw_limit, color='r', linestyle='--', alpha=0.3)
        axes[0].axhline(y=-pitch_limit, color='r', linestyle='--', alpha=0.3)
        axes[0].axhline(y=pitch_limit, color='r', linestyle='--', alpha=0.3)
        
        # 2. Error histogram
        errors = np.sqrt((predictions[:, 0] - targets[:, 0])**2 + (predictions[:, 1] - targets[:, 1])**2)
        axes[1].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1].axvline(np.mean(errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(errors):.2f}°')
        axes[1].set_xlabel('Angular Error (degrees)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Error vs Angle magnitude
        target_magnitudes = np.sqrt(targets[:, 0]**2 + targets[:, 1]**2)
        axes[2].scatter(target_magnitudes, errors, alpha=0.5, s=10)
        axes[2].set_xlabel('Gaze Angle Magnitude (degrees)')
        axes[2].set_ylabel('Prediction Error (degrees)')
        axes[2].set_title('Error vs Gaze Magnitude')
        
        # Add trend line
        if len(target_magnitudes) > 1:
            z = np.polyfit(target_magnitudes, errors, 1)
            p = np.poly1d(z)
            axes[2].plot(sorted(target_magnitudes), p(sorted(target_magnitudes)), 
                        'r--', label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return errors

# ============================================================================
# LOSO CROSS-VALIDATION
# ============================================================================

class MPIIGazeLOSODataset(Dataset):
    """
    Dataset for LOSO cross-validation
    """
    def __init__(self, df, test_subject, val_subjects=None, mode='train'):
        """
        Args:
            df: DataFrame with all data
            test_subject: Subject held out for testing
            val_subjects: Subjects used for validation
            mode: 'train', 'val', or 'test'
        """
        self.df = df.copy()
        self.test_subject = test_subject
        self.val_subjects = val_subjects if val_subjects else []
        self.mode = mode

        # Apply split
        self._apply_split()

        print(f"Created {mode} dataset: {len(self.df)} samples "
              f"(test subject: {test_subject})")

    def _apply_split(self):
        """Apply LOSO split"""
        if self.mode == 'test':
            # Test set: only test subject
            self.df = self.df[self.df['subject'] == self.test_subject].copy()
        elif self.mode == 'val':
            # Validation set: specified validation subjects
            self.df = self.df[self.df['subject'].isin(self.val_subjects)].copy()
        else:
            # Training set: all subjects except test and validation
            exclude = [self.test_subject] + self.val_subjects
            self.df = self.df[~self.df['subject'].isin(exclude)].copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get image
        img = row['image']
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.expand_dims(img, 0)  # Add channel dimension
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img.transpose(2, 0, 1)

        # Get gaze angles
        yaw = float(row['gaze_yaw'])
        pitch = float(row['gaze_pitch'])

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        gaze_tensor = torch.tensor([yaw, pitch], dtype=torch.float32)

        return img_tensor, gaze_tensor

# ============================================================================
# MODIFIED PHYSGAZE TRAINER WITH DYNAMIC EPOCHS
# ============================================================================

class EnhancedPhysGazeTrainer:
    """Enhanced trainer with dynamic epoch stopping and visualization"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Dynamic epoch manager
        self.epoch_manager = DynamicEpochManager(config)
        
        # Visualizer
        self.visualizer = EnhancedVisualizer()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('patience', 5),
            min_lr=1e-6
        )
        
        # History with enhanced tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_outliers': [], 'val_outliers': [],
            'lr': [],
            'train_reg_loss': [], 'train_cycle_loss': [], 'train_acm_loss': []
        }
        
        self.best_val_mae = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_outliers = 0
        total_reg_loss = 0
        total_cycle_loss = 0
        total_acm_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(images, return_all=True)
            gaze_pred = predictions['gaze_corrected']
            
            # Compute losses
            losses = self.model.compute_losses(predictions, targets, images)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mae = F.l1_loss(gaze_pred, targets).item()
                outliers = self.model.get_outlier_rate(gaze_pred)
                
                batch_size = images.size(0)
                total_loss += losses['total'].item() * batch_size
                total_mae += mae * batch_size
                total_outliers += outliers * batch_size
                total_reg_loss += losses['reg'].item() * batch_size
                total_cycle_loss += losses['cycle'].item() * batch_size
                total_acm_loss += losses['acm'].item() * batch_size
                total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mae': f"{mae:.2f}°",
                'outliers': f"{outliers:.1f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        avg_outliers = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
        avg_reg_loss = total_reg_loss / total_samples if total_samples > 0 else 0
        avg_cycle_loss = total_cycle_loss / total_samples if total_samples > 0 else 0
        avg_acm_loss = total_acm_loss / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_mae, avg_outliers, avg_reg_loss, avg_cycle_loss, avg_acm_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_outliers = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_inits = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            predictions = self.model(images, return_all=True)
            gaze_pred = predictions['gaze_corrected']
            
            losses = self.model.compute_losses(predictions, targets, images)
            mae = F.l1_loss(gaze_pred, targets).item()
            outliers = self.model.get_outlier_rate(gaze_pred)
            
            batch_size = images.size(0)
            total_loss += losses['total'].item() * batch_size
            total_mae += mae * batch_size
            total_outliers += outliers * batch_size
            total_samples += batch_size
            
            all_preds.append(gaze_pred.cpu())
            all_targets.append(targets.cpu())
            all_inits.append(predictions['gaze_init'].cpu())
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mae': f"{mae:.2f}°",
                'outliers': f"{outliers:.1f}%"
            })
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        avg_outliers = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
        
        self.scheduler.step(avg_mae)
        
        return avg_loss, avg_mae, avg_outliers, all_preds, all_targets, all_inits
    
    def train(self, max_epochs=100):
        """Train with dynamic epoch stopping"""
        print(f"\n🚀 Starting training with dynamic epoch stopping...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print(f"Max epochs: {max_epochs}")
        print(f"Early stopping patience: {self.epoch_manager.patience}")
        
        for epoch in range(max_epochs):
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_mae, train_outliers, train_reg_loss, train_cycle_loss, train_acm_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_mae, val_outliers, val_preds, val_targets, val_inits = self.validate(epoch)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_outliers'].append(train_outliers)
            self.history['val_outliers'].append(val_outliers)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['train_reg_loss'].append(train_reg_loss)
            self.history['train_cycle_loss'].append(train_cycle_loss)
            self.history['train_acm_loss'].append(train_acm_loss)
            
            # Update epoch manager
            should_stop = self.epoch_manager.update(
                train_loss, val_loss, train_mae, val_mae
            )
            
            # Print epoch summary
            print(f"\n📈 Epoch {epoch + 1} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}°, Outliers: {train_outliers:.1f}%")
            print(f"    Reg: {train_reg_loss:.4f}, Cycle: {train_cycle_loss:.4f}, ACM: {train_acm_loss:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}°, Outliers: {val_outliers:.1f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Calculate improvement
            if epoch > 0:
                val_improvement = ((self.history['val_mae'][-2] - val_mae) / 
                                  self.history['val_mae'][-2] * 100)
                print(f"  Val MAE Improvement: {val_improvement:+.2f}%")
            
            # Save best model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch + 1
                print(f"  🎯 New best model! (MAE: {val_mae:.2f}°)")
            
            # Check if we should stop
            if should_stop:
                print(f"\n🛑 Stopping early after {epoch + 1} epochs due to no improvement")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Loaded best model from epoch {self.best_epoch} (MAE: {self.best_val_mae:.2f}°)")
        
        # Get training summary
        summary = self.epoch_manager.get_training_summary()
        if summary:
            print(f"\n📋 Training Summary:")
            print(f"  Total epochs: {summary['epochs']}")
            print(f"  Best epoch: {self.best_epoch}")
            print(f"  Best val MAE: {self.best_val_mae:.2f}°")
            print(f"  Final train-val gap: {summary['train_val_gap']:.4f}")
            print(f"  Overfitting ratio: {summary['overfitting_ratio']:.2%}")
        
        return self.history

# ============================================================================
# MODIFIED LOSO EVALUATOR WITH DYNAMIC TRAINING
# ============================================================================

class EnhancedLOSOEvaluator:
    """
    Enhanced LOSO evaluator with dynamic epoch stopping and comprehensive visualizations
    """
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.all_subjects = sorted(df['subject'].unique())
        
        # Enhanced configuration
        self.config['early_stopping_patience'] = config.get('early_stopping_patience', 10)
        self.config['early_stopping_min_delta'] = config.get('early_stopping_min_delta', 0.001)
        self.config['overfitting_threshold'] = config.get('overfitting_threshold', 0.1)
        self.config['max_epochs'] = config.get('max_epochs', 100)
        
        print(f"\n🎯 ENHANCED LOSO EVALUATION SETUP")
        print(f"Total subjects: {len(self.all_subjects)}")
        print(f"Subjects: {self.all_subjects}")
        print(f"Dynamic epoch stopping: Enabled")
        print(f"Max epochs per fold: {self.config['max_epochs']}")
        print(f"Early stopping patience: {self.config['early_stopping_patience']}")
        
        # Store results
        self.results = {}
        self.fold_histories = {}
        self.training_summaries = {}
        
        # Visualizer
        self.visualizer = EnhancedVisualizer()
        
    def run(self):
        """Run enhanced LOSO cross-validation"""
        print("\n" + "="*80)
        print("🚀 ENHANCED LOSO CROSS-VALIDATION WITH DYNAMIC EPOCHS")
        print("="*80)
        
        fold_results = {}
        fold_training_dynamics = {}
        
        # Create LOSO folds
        for fold_idx, test_subject in enumerate(self.all_subjects):
            print(f"\n\n{'='*70}")
            print(f"🧪 FOLD {fold_idx + 1}/{len(self.all_subjects)}")
            print(f"Test Subject: {test_subject}")
            print(f"{'='*70}")
            
            # Create datasets
            remaining = [s for s in self.all_subjects if s != test_subject]
            n_val = max(1, int(len(remaining) * 0.2))
            val_subjects = remaining[:n_val]
            train_subjects = remaining[n_val:]
            
            print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
            print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
            
            train_dataset = MPIIGazeLOSODataset(
                self.df, test_subject, val_subjects, mode='train'
            )
            val_dataset = MPIIGazeLOSODataset(
                self.df, test_subject, val_subjects, mode='val'
            )
            test_dataset = MPIIGazeLOSODataset(
                self.df, test_subject, [], mode='test'
            )
            
            # Skip if datasets are empty
            if len(train_dataset) == 0:
                print(f"⚠️ No training data for fold {fold_idx}. Skipping.")
                continue
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            # Train and evaluate this fold with dynamic epochs
            fold_result, fold_history = self._train_and_evaluate_fold_dynamic(
                fold_idx, test_subject,
                train_loader, val_loader, test_loader
            )
            
            if fold_result:
                fold_results[test_subject] = fold_result
                fold_training_dynamics[test_subject] = fold_history
        
        # Calculate overall statistics
        if fold_results:
            self._calculate_enhanced_statistics(fold_results, fold_training_dynamics)
        
        return self.results
    
    def _train_and_evaluate_fold_dynamic(self, fold_idx, test_subject,
                                        train_loader, val_loader, test_loader):
        """Train and evaluate one fold with dynamic epoch stopping"""
        # Create model
        model = PhysGaze(image_size=(36, 60))
        
        # Create enhanced trainer
        trainer = EnhancedPhysGazeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=self.config
        )
        
        # Train with dynamic epochs
        print(f"\n🏋️ Training for fold {fold_idx + 1} (Test: {test_subject})...")
        history = trainer.train(max_epochs=self.config['max_epochs'])
        
        # Store training history
        self.fold_histories[test_subject] = history
        
        # Plot training curves for this fold
        print(f"\n📊 Generating training visualizations for {test_subject}...")
        save_path = f"training_curves_fold_{test_subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.visualizer.plot_fold_training_curves(history, test_subject, save_path)
        
        # Evaluate on test set
        print(f"\n🧪 Evaluating on test subject {test_subject}...")
        test_results = self._evaluate_model_enhanced(model, test_loader, test_subject)
        
        if test_results:
            print(f"\n📊 Results for {test_subject}:")
            print(f"  MAE: {test_results['mae']:.2f}°")
            print(f"  Yaw MAE: {test_results['mae_yaw']:.2f}°")
            print(f"  Pitch MAE: {test_results['mae_pitch']:.2f}°")
            print(f"  Init Outliers: {test_results['init_outlier_rate']:.1f}%")
            print(f"  Final Outliers: {test_results['final_outlier_rate']:.1f}%")
            print(f"  Outlier Reduction: {test_results['outlier_reduction']:.1f}%")
            print(f"  Samples: {test_results['n_samples']:,}")
            print(f"  Training Epochs: {len(history['train_loss'])}")
            
            # Plot gaze vector comparison
            self.visualizer.plot_gaze_vector_comparison(
                test_results['predictions'],
                test_results['targets'],
                test_results['initial_predictions'],
                title=f"Gaze Estimation - {test_subject}"
            )
        
        return test_results, history
    
    def _evaluate_model_enhanced(self, model, test_loader, test_subject):
        """Enhanced model evaluation with more metrics"""
        model.eval()
        
        all_preds = []
        all_targets = []
        all_inits = []
        init_outliers = []
        final_outliers = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f"Testing {test_subject}"):
                images, targets = images.to(device), targets.to(device)
                
                # Get predictions
                predictions = model(images, return_all=True)
                gaze_pred = predictions['gaze_corrected']
                gaze_init = predictions['gaze_init']
                
                # Calculate outliers
                init_outlier = model.get_outlier_rate(gaze_init)
                final_outlier = model.get_outlier_rate(gaze_pred)
                
                # Store results
                all_preds.append(gaze_pred.cpu())
                all_targets.append(targets.cpu())
                all_inits.append(gaze_init.cpu())
                init_outliers.append(init_outlier)
                final_outliers.append(final_outlier)
        
        if not all_preds:
            return None
        
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_inits = torch.cat(all_inits, dim=0)
        
        # Calculate metrics
        mae = F.l1_loss(all_preds, all_targets).item()
        mae_yaw = F.l1_loss(all_preds[:, 0], all_targets[:, 0]).item()
        mae_pitch = F.l1_loss(all_preds[:, 1], all_targets[:, 1]).item()
        
        mse = F.mse_loss(all_preds, all_targets)
        rmse = torch.sqrt(mse).item()
        
        init_outlier_rate = np.mean(init_outliers) * 100
        final_outlier_rate = np.mean(final_outliers) * 100
        
        # Calculate additional metrics
        angular_errors = torch.sqrt(
            torch.sum((all_preds - all_targets) ** 2, dim=1)
        )
        mean_angular_error = torch.mean(angular_errors).item()
        std_angular_error = torch.std(angular_errors).item()
        
        # Calculate accuracy at different thresholds
        thresholds = [5.0, 10.0, 15.0]
        accuracies = {}
        for thresh in thresholds:
            acc = (angular_errors <= thresh).float().mean().item() * 100
            accuracies[f'acc_{int(thresh)}'] = acc
        
        results = {
            'mae': mae,
            'mae_yaw': mae_yaw,
            'mae_pitch': mae_pitch,
            'rmse': rmse,
            'mean_angular_error': mean_angular_error,
            'std_angular_error': std_angular_error,
            'init_outlier_rate': init_outlier_rate,
            'final_outlier_rate': final_outlier_rate,
            'outlier_reduction': ((init_outlier_rate - final_outlier_rate) /
                                 max(init_outlier_rate, 1e-10) * 100),
            'n_samples': len(all_preds),
            'predictions': all_preds.numpy(),
            'targets': all_targets.numpy(),
            'initial_predictions': all_inits.numpy(),
            'angular_errors': angular_errors.numpy(),
            **accuracies
        }
        
        # Store in results
        self.results[test_subject] = results
        
        return results
    
    def _calculate_enhanced_statistics(self, fold_results, fold_training_dynamics):
        """Calculate enhanced statistics with training dynamics analysis"""
        print("\n" + "="*80)
        print("📈 ENHANCED LOSO CROSS-VALIDATION FINAL RESULTS")
        print("="*80)
        
        # Extract metrics
        subjects = list(fold_results.keys())
        maes = [fold_results[s]['mae'] for s in subjects]
        maes_yaw = [fold_results[s]['mae_yaw'] for s in subjects]
        maes_pitch = [fold_results[s]['mae_pitch'] for s in subjects]
        init_outliers = [fold_results[s]['init_outlier_rate'] for s in subjects]
        final_outliers = [fold_results[s]['final_outlier_rate'] for s in subjects]
        outlier_reductions = [fold_results[s]['outlier_reduction'] for s in subjects]
        samples = [fold_results[s]['n_samples'] for s in subjects]
        training_epochs = [len(fold_training_dynamics[s]['train_loss']) for s in subjects]
        
        # Calculate statistics
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        mean_mae_yaw = np.mean(maes_yaw)
        std_mae_yaw = np.std(maes_yaw)
        mean_mae_pitch = np.mean(maes_pitch)
        std_mae_pitch = np.std(maes_pitch)
        
        mean_init_outliers = np.mean(init_outliers)
        mean_final_outliers = np.mean(final_outliers)
        mean_outlier_reduction = np.mean(outlier_reductions)
        mean_training_epochs = np.mean(training_epochs)
        
        # Print results
        print(f"\n📊 Overall Performance:")
        print(f"  Mean MAE: {mean_mae:.2f}° ± {std_mae:.2f}°")
        print(f"  Mean Yaw MAE: {mean_mae_yaw:.2f}° ± {std_mae_yaw:.2f}°")
        print(f"  Mean Pitch MAE: {mean_mae_pitch:.2f}° ± {std_mae_pitch:.2f}°")
        
        print(f"\n👁️ Outlier Analysis:")
        print(f"  Initial Outlier Rate: {mean_init_outliers:.1f}%")
        print(f"  Final Outlier Rate: {mean_final_outliers:.1f}%")
        print(f"  Mean Outlier Reduction: {mean_outlier_reduction:.1f}%")
        
        print(f"\n⏱️ Training Efficiency:")
        print(f"  Mean Training Epochs: {mean_training_epochs:.1f}")
        print(f"  Epoch Range: [{min(training_epochs)}, {max(training_epochs)}]")
        
        print(f"\n📈 Range and Distribution:")
        print(f"  MAE Range: [{min(maes):.2f}°, {max(maes):.2f}°]")
        print(f"  Best subject: {subjects[np.argmin(maes)]} ({min(maes):.2f}°)")
        print(f"  Worst subject: {subjects[np.argmax(maes)]} ({max(maes):.2f}°)")
        print(f"  Total test samples: {sum(samples):,}")
        
        # Create enhanced visualizations
        self._plot_enhanced_results(subjects, maes, init_outliers, final_outliers, 
                                   training_epochs, fold_training_dynamics)
        
        # Compare with paper
        print(f"\n" + "="*80)
        print("📊 COMPARISON WITH PHYSGAZE PAPER")
        print("="*80)
        print(f"{'Metric':<25} {'Paper (MPIIGaze)':<20} {'Our Implementation':<20}")
        print(f"{'-'*65}")
        print(f"{'Mean Angular Error':<25} {'4.3°':<20} {mean_mae:.2f}° ± {std_mae:.2f}°")
        print(f"{'Outlier Rate':<25} {'0.0%':<20} {mean_final_outliers:.1f}%")
        print(f"{'Training Efficiency':<25} {'50 epochs':<20} {mean_training_epochs:.1f} epochs")
        
        # Create summary table
        summary_data = []
        for subject in subjects:
            results = fold_results[subject]
            summary_data.append({
                'Subject': subject,
                'MAE (°)': f"{results['mae']:.2f}",
                'Yaw MAE (°)': f"{results['mae_yaw']:.2f}",
                'Pitch MAE (°)': f"{results['mae_pitch']:.2f}",
                'Init Outliers (%)': f"{results['init_outlier_rate']:.1f}",
                'Final Outliers (%)': f"{results['final_outlier_rate']:.1f}",
                'Outlier Reduction (%)': f"{results['outlier_reduction']:.1f}",
                'Training Epochs': len(fold_training_dynamics[subject]['train_loss']),
                'Samples': results['n_samples']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(f"\n📋 Detailed Results per Subject:")
        print(summary_df.to_string(index=False))
        
        # Save results
        self._save_enhanced_results(summary_df, fold_results, fold_training_dynamics, 
                                   mean_mae, std_mae, mean_training_epochs)
    
    def _plot_enhanced_results(self, subjects, maes, init_outliers, final_outliers, 
                              training_epochs, fold_training_dynamics):
        """Plot enhanced results with training dynamics"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Enhanced LOSO Analysis with Dynamic Epoch Training', 
                    fontsize=16, fontweight='bold')
        
        x_pos = np.arange(len(subjects))
        
        # 1. MAE vs Training Epochs
        scatter = axes[0, 0].scatter(training_epochs, maes, c=maes, cmap='viridis', 
                                     s=100, alpha=0.7, edgecolors='black')
        axes[0, 0].set_xlabel('Training Epochs')
        axes[0, 0].set_ylabel('MAE (°)')
        axes[0, 0].set_title('MAE vs Training Epochs')
        
        # Add subject labels
        for i, subject in enumerate(subjects):
            axes[0, 0].annotate(subject, (training_epochs[i], maes[i]), 
                              fontsize=8, alpha=0.7)
        
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='MAE (°)')
        
        # 2. Training Efficiency
        axes[0, 1].bar(x_pos, training_epochs, alpha=0.7, color='orange')
        axes[0, 1].axhline(y=np.mean(training_epochs), color='red', linestyle='--',
                          label=f'Mean: {np.mean(training_epochs):.1f}')
        axes[0, 1].set_xlabel('Subject')
        axes[0, 1].set_ylabel('Training Epochs')
        axes[0, 1].set_title('Training Efficiency per Fold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(subjects, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Outlier Reduction
        width = 0.35
        axes[0, 2].bar(x_pos - width/2, init_outliers, width, label='Initial', alpha=0.7, color='red')
        axes[0, 2].bar(x_pos + width/2, final_outliers, width, label='After ACM', alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Subject')
        axes[0, 2].set_ylabel('Outlier Rate (%)')
        axes[0, 2].set_title('Outlier Reduction by ACM')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(subjects, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training Dynamics Heatmap (Loss convergence)
        if len(fold_training_dynamics) > 0:
            max_epochs = max([len(d['val_loss']) for d in fold_training_dynamics.values()])
            loss_matrix = np.zeros((len(subjects), max_epochs))
            
            for i, subject in enumerate(subjects):
                val_loss = fold_training_dynamics[subject]['val_loss']
                loss_matrix[i, :len(val_loss)] = val_loss
                if len(val_loss) < max_epochs:
                    loss_matrix[i, len(val_loss):] = np.nan
            
            im = axes[1, 0].imshow(loss_matrix, aspect='auto', cmap='hot_r', 
                                  interpolation='nearest')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Subject')
            axes[1, 0].set_title('Validation Loss Heatmap')
            axes[1, 0].set_yticks(range(len(subjects)))
            axes[1, 0].set_yticklabels(subjects)
            plt.colorbar(im, ax=axes[1, 0], label='Validation Loss')
        
        # 5. Convergence Speed Analysis
        convergence_data = []
        for subject in subjects:
            val_loss = fold_training_dynamics[subject]['val_loss']
            if len(val_loss) > 0:
                # Find epoch where loss is within 10% of final loss
                final_loss = val_loss[-1]
                target_loss = final_loss * 1.1
                convergence_epoch = next((i for i, loss in enumerate(val_loss) 
                                         if loss <= target_loss), len(val_loss))
                convergence_data.append(convergence_epoch)
        
        if convergence_data:
            axes[1, 1].bar(range(len(convergence_data)), convergence_data, 
                          alpha=0.7, color='purple')
            axes[1, 1].set_xlabel('Subject Index')
            axes[1, 1].set_ylabel('Convergence Epoch')
            axes[1, 1].set_title('Model Convergence Speed')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Overfitting Analysis per Fold
        overfitting_gaps = []
        for subject in subjects:
            history = fold_training_dynamics[subject]
            if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
                gap = history['val_loss'][-1] - history['train_loss'][-1]
                overfitting_gaps.append(gap)
        
        axes[1, 2].bar(x_pos, overfitting_gaps, alpha=0.7, 
                      color=['red' if gap > 0.1 else 'green' for gap in overfitting_gaps])
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Overfitting threshold')
        axes[1, 2].set_xlabel('Subject')
        axes[1, 2].set_ylabel('Val Loss - Train Loss')
        axes[1, 2].set_title('Overfitting Gap per Fold')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(subjects, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. MAE Distribution
        axes[2, 0].hist(maes, bins=15, alpha=0.7, color='blue', edgecolor='black')
        axes[2, 0].axvline(np.mean(maes), color='red', linestyle='--',
                          label=f'Mean: {np.mean(maes):.2f}°')
        axes[2, 0].axvline(np.median(maes), color='green', linestyle='--',
                          label=f'Median: {np.median(maes):.2f}°')
        axes[2, 0].set_xlabel('MAE (°)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Distribution of MAE Across Subjects')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Correlation: Training Epochs vs MAE
        if len(training_epochs) == len(maes):
            axes[2, 1].scatter(training_epochs, maes, alpha=0.7, s=80, edgecolors='black')
            
            # Add trend line
            if len(training_epochs) > 1:
                z = np.polyfit(training_epochs, maes, 1)
                p = np.poly1d(z)
                axes[2, 1].plot(sorted(training_epochs), p(sorted(training_epochs)), 
                               'r--', label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
            
            axes[2, 1].set_xlabel('Training Epochs')
            axes[2, 1].set_ylabel('MAE (°)')
            axes[2, 1].set_title('Correlation: Epochs vs MAE')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        axes[2, 2].axis('off')
        summary_text = (
            f"Summary Statistics:\n"
            f"Total Subjects: {len(subjects)}\n"
            f"Mean MAE: {np.mean(maes):.2f}°\n"
            f"Std MAE: {np.std(maes):.2f}°\n"
            f"Mean Training Epochs: {np.mean(training_epochs):.1f}\n"
            f"Mean Outlier Reduction: {np.mean(final_outliers):.1f}%\n"
            f"Best Subject: {subjects[np.argmin(maes)]}\n"
            f"Worst Subject: {subjects[np.argmax(maes)]}"
        )
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def _save_enhanced_results(self, summary_df, fold_results, fold_training_dynamics,
                              mean_mae, std_mae, mean_training_epochs):
        """Save enhanced results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary as CSV
        csv_path = f"enhanced_physgaze_loso_results_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved summary to: {csv_path}")
        
        # Save detailed results as JSON
        results_dict = {
            'timestamp': timestamp,
            'config': self.config,
            'overall_metrics': {
                'mean_mae': float(mean_mae),
                'std_mae': float(std_mae),
                'mean_training_epochs': float(mean_training_epochs),
                'n_subjects': len(fold_results),
                'total_samples': int(summary_df['Samples'].sum())
            },
            'per_subject_metrics': {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                                       for kk, vv in v.items() if kk not in ['predictions', 'targets', 'initial_predictions', 'angular_errors']}
                                   for k, v in fold_results.items()},
            'training_dynamics': {k: {kk: ([float(x) for x in vv] if isinstance(vv, list) else float(vv))
                                     for kk, vv in v.items()}
                                 for k, v in fold_training_dynamics.items()}
        }
        
        json_path = f"enhanced_physgaze_loso_detailed_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"💾 Saved detailed results to: {json_path}")

# ============================================================================
# MODIFIED MAIN FUNCTION
# ============================================================================

def main_enhanced():
    """Enhanced main function with dynamic epoch training"""
    print("="*80)
    print("👁️ ENHANCED PHYSGAZE: LOSO EVALUATION WITH DYNAMIC EPOCHS")
    print("="*80)
    
    # Enhanced configuration
    config = {
        # Data
        'data_dir': './data/MPIIGaze/MPIIGaze',
        
        # Training with dynamic epochs
        'batch_size': 128,
        'max_epochs': 50,           # Reduced for testing - can increase to 100 for full training
        'early_stopping_patience': 5, # Reduced for testing
        'early_stopping_min_delta': 0.001,
        'overfitting_threshold': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'patience': 3,  # Reduced for testing
        
        # Model
        'image_size': (36, 60),
    }
    
    print("\n📋 Enhanced Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load MPIIGaze dataset
    print("\n📁 Loading MPIIGaze dataset...")
    df = MPIIGazeLoader.load_all_subjects(base_dir=config['data_dir'])
    
    # Run enhanced LOSO evaluation
    print("\n" + "="*80)
    print("🚀 STARTING ENHANCED LOSO CROSS-VALIDATION")
    print("="*80)
    
    evaluator = EnhancedLOSOEvaluator(df, config)
    results = evaluator.run()
    
    print("\n" + "="*80)
    print("🎉 ENHANCED LOSO EVALUATION COMPLETED!")
    print("="*80)
    
    # Summary of improvements
    if hasattr(evaluator, 'fold_histories'):
        total_epochs_saved = 0
        original_epochs = 15 * 50  # 15 subjects * 50 epochs each
        actual_epochs = sum([len(h['train_loss']) for h in evaluator.fold_histories.values()])
        
        print(f"\n⚡ Training Efficiency Improvements:")
        print(f"  Original (fixed 50 epochs): {original_epochs} total epochs")
        print(f"  Actual (dynamic stopping): {actual_epochs} total epochs")
        print(f"  Epochs saved: {original_epochs - actual_epochs} ({((original_epochs - actual_epochs)/original_epochs*100):.1f}% reduction)")
        print(f"  Average epochs per fold: {actual_epochs/len(evaluator.fold_histories):.1f}")

# ============================================================================
# RUN ENHANCED VERSION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 ENHANCED PHYSGAZE LOSO EVALUATION WITH DYNAMIC EPOCHS")
    print("="*80)
    
    # Choose which version to run
    use_enhanced = True
    
    if use_enhanced:
        main_enhanced()
    else:
        print("Original version not available. Running enhanced version...")
        main_enhanced()