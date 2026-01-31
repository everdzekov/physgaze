import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import h5py
import scipy.io as sio
from pathlib import Path
import urllib.request
import tarfile
import shutil
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 2: Dataset Download and Preprocessing
# ============================================================================

class MPIIGazeDataset(Dataset):
    """
    MPIIGaze Dataset for gaze estimation.

    The dataset contains 213,659 images from 15 subjects collected in
    everyday laptop use over several months.

    Reference: Zhang et al., "Appearance-Based Gaze Estimation in the Wild" (CVPR 2015)
    """

    def __init__(
        self,
        root_dir: str = './data/MPIIGaze',
        split: str = 'train',
        subject_ids: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        normalize_gaze: bool = True,
        debug: bool = True  # Added debug flag
    ):
        """
        Args:
            root_dir: Root directory of the MPIIGaze dataset
            split: 'train', 'val', or 'test'
            subject_ids: List of subject IDs to include (0-14)
            transform: Optional transforms to apply
            normalize_gaze: Whether to normalize gaze angles
            debug: Enable debug print statements
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.normalize_gaze = normalize_gaze
        self.debug = debug

        # Biomechanical constraints (degrees)
        self.yaw_max = 55.0
        self.pitch_max = 40.0

        # Default subject split
        if subject_ids is None:
            if split == 'train':
                subject_ids = list(range(0, 12))  # p00-p11
            elif split == 'val':
                subject_ids = [12, 13]  # p12, p13
            else:  # test
                subject_ids = [14]  # p14

        self.subject_ids = subject_ids
        self.samples = []

        if self.debug:
            print("\n" + "="*60)
            print("MPIIGazeDataset INITIALIZATION DEBUG")
            print("="*60)
            print(f"Root directory: {self.root_dir}")
            print(f"Absolute path: {self.root_dir.absolute()}")
            print(f"Root exists: {self.root_dir.exists()}")
            print(f"Split: {self.split}")
            print(f"Subject IDs: {self.subject_ids}")
            print(f"Normalize gaze: {self.normalize_gaze}")
            print("-"*60)

        # Load dataset
        self._load_dataset()

        if self.debug:
            print(f"\nTotal samples loaded: {len(self.samples)}")
            if len(self.samples) > 0:
                print(f"First sample keys: {list(self.samples[0].keys())}")
                if 'image' in self.samples[0]:
                    print(f"First image shape: {self.samples[0]['image'].shape if hasattr(self.samples[0]['image'], 'shape') else 'Unknown'}")
                if 'gaze' in self.samples[0]:
                    print(f"First gaze shape: {self.samples[0]['gaze'].shape if hasattr(self.samples[0]['gaze'], 'shape') else 'Unknown'}")
            print("="*60)

    def _load_dataset(self):
        """Load the dataset from processed files or raw data."""
        processed_file = self.root_dir / f'processed_{self.split}.pt'

        if self.debug:
            print(f"\n[LOAD DATASET]")
            print(f"Looking for processed file: {processed_file}")
            print(f"Processed file exists: {processed_file.exists()}")

        if processed_file.exists():
            print(f"Loading processed {self.split} data...")
            try:
                data = torch.load(processed_file)
                self.samples = data['samples']
                if self.debug:
                    print(f"Successfully loaded {len(self.samples)} samples from processed file")
            except Exception as e:
                print(f"Error loading processed file: {e}")
                print("Falling back to raw data processing...")
                self._process_raw_data()
        else:
            if self.debug:
                print(f"No processed file found, processing raw data...")
            self._process_raw_data()

    def _process_raw_data(self):
        """Process raw MPIIGaze data."""
        if self.debug:
            print(f"\n[PROCESS RAW DATA]")
            print(f"Processing subjects: {self.subject_ids}")

        total_samples_before = len(self.samples)
        total_subjects_processed = 0
        total_files_processed = 0
        total_errors = 0

        for subject_id in self.subject_ids:
            subject_dir = self.root_dir / 'Data' / 'Normalized' / f'p{subject_id:02d}'

            if self.debug:
                print(f"\n  [SUBJECT p{subject_id:02d}]")
                print(f"  Subject directory: {subject_dir}")
                print(f"  Directory exists: {subject_dir.exists()}")

            if not subject_dir.exists():
                print(f"  Warning: Subject directory {subject_dir} not found")
                if self.debug:
                    print(f"  Full path: {subject_dir.absolute()}")
                    # List parent directory contents
                    parent_dir = subject_dir.parent
                    if parent_dir.exists():
                        print(f"  Parent directory exists: {[f.name for f in parent_dir.glob('*')]}")
                continue

            # List all files in subject directory
            all_files = list(subject_dir.glob('*'))
            if self.debug:
                print(f"  Total files in directory: {len(all_files)}")
                if len(all_files) > 0:
                    print(f"  First few files: {[f.name for f in all_files[:5]]}")

            # Load all .mat files for this subject
            mat_files = list(subject_dir.glob('*.mat'))

            if self.debug:
                print(f"  Found {len(mat_files)} .mat files")
                if len(mat_files) > 0:
                    print(f"  First .mat file: {mat_files[0].name}")

            if len(mat_files) == 0:
                print(f"  Warning: No .mat files found for subject p{subject_id:02d}")
                # Try other extensions
                for ext in ['*.pkl', '*.h5', '*.npz', '*.npy', '*.txt']:
                    other_files = list(subject_dir.glob(ext))
                    if other_files:
                        print(f"  Found {len(other_files)} {ext} files instead")
                        break
                continue

            subject_samples_before = len(self.samples)
            subject_files_processed = 0

            for mat_file in mat_files:
                if self.debug:
                    print(f"    Processing file: {mat_file.name}")

                try:
                    mat_data = sio.loadmat(str(mat_file))

                    if self.debug:
                        print(f"      Keys in .mat file: {list(mat_data.keys())}")
                        print(f"      File size: {mat_file.stat().st_size / 1024:.2f} KB")

                    # Extract data based on MPIIGaze format
                    if 'data' in mat_data:
                        data = mat_data['data']

                        if self.debug:
                            print(f"      Data shape: {data.shape}")
                            print(f"      Data type: {type(data)}")

                        for i in range(data.shape[1]):
                            sample = data[0, i]

                            # Check if 'left' and 'right' exist
                            if 'left' not in sample.dtype.names or 'right' not in sample.dtype.names:
                                if self.debug:
                                    print(f"      Warning: Sample {i} missing 'left' or 'right' keys in {mat_file.name}")
                                continue

                            # Extract raw gaze data for left eye
                            raw_left_gaze = sample['left'][0, 0]['gaze'][0, 0]
                            # Validate gaze data: ensure it's a 3D vector
                            if not isinstance(raw_left_gaze, np.ndarray) or raw_left_gaze.size != 3:
                                if self.debug:
                                    print(f"      Warning: Left gaze for sample {i} in {mat_file.name} is not a 3D vector (shape: {raw_left_gaze.shape if isinstance(raw_left_gaze, np.ndarray) else 'scalar'}), skipping this eye.")
                                continue
                            left_eye = sample['left'][0, 0]['image'][0, 0]
                            left_gaze = raw_left_gaze.flatten()

                            # Extract raw gaze data for right eye
                            raw_right_gaze = sample['right'][0, 0]['gaze'][0, 0]
                            # Validate gaze data: ensure it's a 3D vector
                            if not isinstance(raw_right_gaze, np.ndarray) or raw_right_gaze.size != 3:
                                if self.debug:
                                    print(f"      Warning: Right gaze for sample {i} in {mat_file.name} is not a 3D vector (shape: {raw_right_gaze.shape if isinstance(raw_right_gaze, np.ndarray) else 'scalar'}), skipping this eye.")
                                continue
                            right_eye = sample['right'][0, 0]['image'][0, 0]
                            right_gaze = raw_right_gaze.flatten()

                            if self.debug and i == 0 and subject_files_processed == 0:
                                print(f"      First sample - left eye shape: {left_eye.shape}")
                                print(f"      First sample - left gaze shape: {left_gaze.shape}")
                                print(f"      First sample - right eye shape: {right_eye.shape}")
                                print(f"      First sample - right gaze shape: {right_gaze.shape}")

                            # Add both eyes as separate samples
                            self.samples.append({
                                'image': left_eye,
                                'gaze': left_gaze,
                                'subject_id': subject_id
                            })
                            self.samples.append({
                                'image': right_eye,
                                'gaze': right_gaze,
                                'subject_id': subject_id
                            })

                        subject_files_processed += 1
                        total_files_processed += 1

                    else:
                        if self.debug:
                            print(f"      Warning: 'data' key not found in {mat_file.name}")
                        total_errors += 1

                except Exception as e:
                    print(f"    Error loading {mat_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    total_errors += 1
                    continue

            subject_samples_added = len(self.samples) - subject_samples_before
            if self.debug:
                print(f"  Added {subject_samples_added} samples from subject p{subject_id:02d}")

            if subject_samples_added > 0:
                total_subjects_processed += 1

        # Summary
        if self.debug:
            print(f"\n[PROCESSING SUMMARY]")
            print(f"  Total subjects to process: {len(self.subject_ids)}")
            print(f"  Subjects successfully processed: {total_subjects_processed}")
            print(f"  Files successfully processed: {total_files_processed}")
            print(f"  Total errors encountered: {total_errors}")
            print(f"  Total samples before: {total_samples_before}")
            print(f"  Total samples after: {len(self.samples)}")
            print(f"  Samples added: {len(self.samples) - total_samples_before}")

        # Save processed data if we have samples
        if len(self.samples) > 0:
            processed_file = self.root_dir / f'processed_{self.split}.pt'
            try:
                torch.save({'samples': self.samples}, processed_file)
                if self.debug:
                    print(f"  Saved {len(self.samples)} samples to {processed_file}")
            except Exception as e:
                print(f"  Error saving processed file: {e}")
        else:
            print(f"  Warning: No samples were loaded!")

            # Additional diagnostics
            if self.debug:
                print(f"\n[ADDITIONAL DIAGNOSTICS]")
                print(f"  Checking root directory structure...")

                # Check if root exists
                if not self.root_dir.exists():
                    print(f"  Root directory does not exist!")
                    print(f"  Create the directory or download the dataset to: {self.root_dir.absolute()}")
                else:
                    print(f"  Root directory exists")

                    # List all contents recursively
                    print(f"  Contents of root directory:")
                    for item in self.root_dir.rglob('*'):
                        rel_path = item.relative_to(self.root_dir)
                        if item.is_file():
                            print(f"    File: {rel_path}")
                        else:
                            print(f"    Dir:  {rel_path}/")

    def __len__(self) -> int:
        if self.debug:
            print(f"[__len__] called: returning {len(self.samples)}")
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.debug:
            print(f"[__getitem__] called with idx={idx}, total samples={len(self.samples)}")

        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.samples)} samples")

        sample = self.samples[idx]

        image = sample['image']
        gaze = sample['gaze']

        if self.debug and idx == 0:
            print(f"  Sample type: {type(sample)}")
            print(f"  Image type: {type(image)}, shape: {image.shape if hasattr(image, 'shape') else 'No shape'}")
            print(f"  Gaze type: {type(gaze)}, shape: {gaze.shape if hasattr(gaze, 'shape') else 'No shape'}")

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        elif not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Ensure image is [C, H, W] format
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[-1] in [1, 3]:
            image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        elif self.debug and idx == 0:
            print(f"  Image shape after processing: {image.shape}")

        # Normalize image to [0, 1]
        if image.max() > 1:
            image = image / 255.0

        # Convert gaze to spherical coordinates (yaw, pitch) in degrees
        if isinstance(gaze, np.ndarray):
            gaze_tensor = torch.from_numpy(gaze).float()
        else:
            gaze_tensor = torch.tensor(gaze, dtype=torch.float32)

        # Convert from 3D gaze vector to spherical if needed
        if len(gaze_tensor) == 3:
            # gaze = [x, y, z] -> (yaw, pitch)
            yaw = torch.atan2(gaze_tensor[0], -gaze_tensor[2]) * 180 / np.pi
            pitch = torch.asin(gaze_tensor[1]) * 180 / np.pi
            gaze_tensor = torch.stack([yaw, pitch])
        # If gaze_tensor is already 2D (yaw, pitch) or has unexpected length, this part needs care.
        # Based on previous debug output, gaze_tensor might be (1,) here, which is an issue.
        # The fix above in _process_raw_data aims to prevent (1,) from reaching here.
        # If it's 2D and not 3D, it implies it's already yaw/pitch, which is handled directly by normalization.

        # Normalize gaze angles to [-1, 1]
        if self.normalize_gaze:
            # This is where the IndexError occurred when gaze_tensor was (1,)
            # The check in _process_raw_data should prevent this case.
            gaze_tensor[0] = gaze_tensor[0] / self.yaw_max
            gaze_tensor[1] = gaze_tensor[1] / self.pitch_max
            gaze_tensor = torch.clamp(gaze_tensor, -1, 1)

        if self.transform:
            image = self.transform(image)

        return image, gaze_tensor

class SyntheticMPIIGaze(Dataset):
    """
    Synthetic MPIIGaze-like dataset for testing without downloading the real data.
    Generates realistic eye images with known gaze directions.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: Tuple[int, int] = (36, 60),
        split: str = 'train',
        normalize_gaze: bool = True
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split
        self.normalize_gaze = normalize_gaze

        # Biomechanical constraints
        self.yaw_max = 55.0
        self.pitch_max = 40.0

        # Generate samples
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Dict]:
        """Generate synthetic eye images with gaze labels."""
        samples = []

        for _ in range(self.num_samples):
            # Sample gaze angles from realistic distribution
            yaw = np.random.uniform(-50, 50)  # degrees
            pitch = np.random.uniform(-35, 25)  # degrees

            # Generate synthetic eye image
            image = self._render_eye(yaw, pitch)

            samples.append({
                'image': image,
                'gaze': np.array([yaw, pitch])
            })

        return samples

    def _render_eye(self, yaw: float, pitch: float) -> np.ndarray:
        """Render a synthetic eye image based on gaze direction."""
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.float32)

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2

        # Eye white (sclera) - ellipse
        sclera_a, sclera_b = w // 3, h // 3
        sclera_mask = ((x - cx) / sclera_a) ** 2 + ((y - cy) / sclera_b) ** 2 <= 1
        image[sclera_mask] = 0.9 + np.random.uniform(-0.05, 0.05)

        # Iris position based on gaze
        iris_offset_x = (yaw / self.yaw_max) * (w // 6)
        iris_offset_y = (pitch / self.pitch_max) * (h // 6)
        iris_cx = cx + iris_offset_x
        iris_cy = cy + iris_offset_y

        # Iris
        iris_r = h // 4
        iris_mask = (x - iris_cx) ** 2 + (y - iris_cy) ** 2 <= iris_r ** 2
        iris_mask = iris_mask & sclera_mask

        # Iris color gradient
        iris_dist = np.sqrt((x - iris_cx) ** 2 + (y - iris_cy) ** 2)
        iris_color = 0.3 + 0.2 * (iris_dist / iris_r)
        image[iris_mask] = np.clip(iris_color[iris_mask], 0.2, 0.5)

        # Pupil
        pupil_r = h // 8
        pupil_mask = (x - iris_cx) ** 2 + (y - iris_cy) ** 2 <= pupil_r ** 2
        pupil_mask = pupil_mask & sclera_mask
        image[pupil_mask] = 0.05

        # Specular highlight
        highlight_cx = iris_cx + 2
        highlight_cy = iris_cy - 2
        highlight_r = 2
        highlight_mask = (x - highlight_cx) ** 2 + (y - highlight_cy) ** 2 <= highlight_r ** 2
        highlight_mask = highlight_mask & iris_mask
        image[highlight_mask] = 0.95

        # Add noise
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)

        return image

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        image = torch.from_numpy(sample['image']).float().unsqueeze(0)
        gaze = torch.from_numpy(sample['gaze']).float()

        if self.normalize_gaze:
            gaze[0] = gaze[0] / self.yaw_max
            gaze[1] = gaze[1] / self.pitch_max

        return image, gaze


def download_mpiigaze(root_dir: str = './data/MPIIGaze') -> bool:
    """
    Download and extract the MPIIGaze dataset.

    Args:
        root_dir: Directory to save the dataset

    Returns:
        True if successful, False otherwise
    """
    url = "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz"
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    tar_file = root_path / "MPIIGaze.tar.gz"

    if (root_path / 'Data').exists():
        print("MPIIGaze dataset already exists.")
        return True

    print(f"Downloading MPIIGaze dataset from {url}...")
    print("This may take a while (~2GB)...")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rDownload progress: {percent:.1f}%", end='')

        urllib.request.urlretrieve(url, tar_file, reporthook=report_progress)
        print("\nDownload complete!")

        # Extract
        print("Extracting dataset...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(root_path)

        # Clean up
        os.remove(tar_file)
        print("Extraction complete!")

        return True

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("You can manually download from:")
        print(url)
        return False


# ============================================================================
# CELL 3: PhysGaze Model Architecture
# ============================================================================

class ResNetBackbone(nn.Module):
    """
    ResNet-18 based backbone for feature extraction.
    Adapted for single-channel eye images.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-18
        resnet = resnet18(pretrained=pretrained)

        # Modify first conv layer for single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize with average of pretrained weights
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)
                )

        # Use remaining ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Initial gaze regression head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # (yaw, pitch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input eye image [B, 1, H, W]

        Returns:
            Initial gaze prediction [B, 2] (yaw, pitch)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        gaze = self.fc(x)
        return gaze


class AnatomicalConstraintModule(nn.Module):
    """
    Anatomical Constraint Module (ACM).

    Projects raw network predictions onto a learned manifold of physiologically
    feasible eye rotations. The ACM ensures predictions stay within biomechanical
    limits (±55° yaw, ±40° pitch).
    """

    def __init__(
        self,
        yaw_max: float = 55.0,
        pitch_max: float = 40.0,
        hidden_dims: List[int] = [64, 32]
    ):
        super().__init__()

        self.yaw_max = yaw_max
        self.pitch_max = pitch_max

        # MLP for learned constraint projection
        layers = []
        in_dim = 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))

        self.mlp = nn.Sequential(*layers)

        # Learnable constraint boundaries (initialized to biomechanical limits)
        self.register_buffer('yaw_limit', torch.tensor(yaw_max))
        self.register_buffer('pitch_limit', torch.tensor(pitch_max))

    def forward(self, gaze_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply anatomical constraints to gaze prediction.

        Args:
            gaze_init: Initial gaze prediction [B, 2] in normalized [-1, 1]

        Returns:
            gaze_valid: Constrained gaze prediction [B, 2]
            correction: Amount of correction applied [B, 2]
        """
        # Apply learned projection
        residual = self.mlp(gaze_init)

        # Soft clamping with tanh (ensures output is in [-1, 1])
        gaze_valid = torch.tanh(gaze_init + residual)

        # Calculate correction magnitude
        correction = gaze_valid - gaze_init

        return gaze_valid, correction

    def is_anatomically_valid(self, gaze: torch.Tensor) -> torch.Tensor:
        """
        Check if gaze predictions are within anatomical limits.

        Args:
            gaze: Gaze prediction [B, 2] in normalized [-1, 1]

        Returns:
            valid: Boolean tensor [B] indicating validity
        """
        # Convert to degrees
        yaw_deg = gaze[:, 0] * self.yaw_max
        pitch_deg = gaze[:, 1] * self.pitch_max

        # Check bounds (with small margin)
        outlier_yaw = 60.0  # degrees
        outlier_pitch = 45.0  # degrees

        valid = (torch.abs(yaw_deg) <= outlier_yaw) & (torch.abs(pitch_deg) <= outlier_pitch)
        return valid


class DifferentiableRenderer(nn.Module):
    """
    Differentiable Renderer for cycle-consistency.

    Decodes predicted gaze direction back into a synthetic eye image,
    enabling cycle-consistency loss that enforces geometric coherence.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (36, 60),
        texture_size: int = 64
    ):
        super().__init__()

        self.image_size = image_size
        self.texture_size = texture_size

        # Learnable texture map (generic eye appearance)
        self.texture = nn.Parameter(
            torch.randn(1, texture_size, texture_size) * 0.1 + 0.5
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, image_size[0] * image_size[1]),
            nn.Sigmoid()
        )

        # Convolutional refinement
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, gaze: torch.Tensor) -> torch.Tensor:
        """
        Render synthetic eye image from gaze direction.

        Args:
            gaze: Gaze direction [B, 2] in normalized [-1, 1]

        Returns:
            rendered: Synthetic eye image [B, 1, H, W]
        """
        batch_size = gaze.shape[0]
        h, w = self.image_size

        # Decode gaze to image
        x = self.decoder(gaze)
        x = x.view(batch_size, 1, h, w)

        # Refine with convolutions
        rendered = self.refine(x)

        return rendered


class PhysGaze(nn.Module):
    """
    PhysGaze: Physics-Informed Deep Learning Framework for Gaze Estimation.

    Combines:
    1. ResNet-18 backbone for feature extraction
    2. Anatomical Constraint Module (ACM) for physiological plausibility
    3. Differentiable Renderer (DR) for cycle-consistency
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        use_acm: bool = True,
        use_renderer: bool = True,
        image_size: Tuple[int, int] = (36, 60)
    ):
        super().__init__()

        self.use_acm = use_acm
        self.use_renderer = use_renderer
        self.image_size = image_size

        # Feature extraction backbone
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)

        # Anatomical Constraint Module
        if use_acm:
            self.acm = AnatomicalConstraintModule()

        # Differentiable Renderer
        if use_renderer:
            self.renderer = DifferentiableRenderer(image_size=image_size)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PhysGaze.

        Args:
            x: Input eye image [B, 1, H, W]
            return_all: Whether to return all intermediate outputs

        Returns:
            Dictionary containing:
            - gaze: Final gaze prediction [B, 2]
            - gaze_init: Initial backbone prediction [B, 2]
            - correction: ACM correction amount [B, 2]
            - rendered: Rendered image from DR [B, 1, H, W]
        """
        # Resize input if needed
        if x.shape[2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)

        # Feature extraction
        gaze_init = self.backbone(x)

        # Apply anatomical constraints
        if self.use_acm:
            gaze, correction = self.acm(gaze_init)
        else:
            gaze = gaze_init
            correction = torch.zeros_like(gaze_init)

        outputs = {
            'gaze': gaze,
            'gaze_init': gaze_init,
            'correction': correction
        }

        # Render synthetic image for cycle-consistency
        if self.use_renderer and return_all:
            rendered = self.renderer(gaze)
            outputs['rendered'] = rendered

        return outputs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Simple prediction interface."""
        outputs = self.forward(x, return_all=False)
        return outputs['gaze']


# ============================================================================
# CELL 4: Loss Functions
# ============================================================================

class PhysGazeLoss(nn.Module):
    """
    Combined loss function for PhysGaze.

    L_total = λ_reg * L_reg + λ_cycle * L_cycle + λ_acm * L_acm

    Where:
    - L_reg: Mean Absolute Error regression loss
    - L_cycle: Cycle-consistency loss (L1 between input and rendered)
    - L_acm: ACM regularization (encourages minimal corrections)
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        lambda_cycle: float = 0.2,
        lambda_acm: float = 0.1,
        yaw_max: float = 55.0,
        pitch_max: float = 40.0
    ):
        super().__init__()

        self.lambda_reg = lambda_reg
        self.lambda_cycle = lambda_cycle
        self.lambda_acm = lambda_acm
        self.yaw_max = yaw_max
        self.pitch_max = pitch_max

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.

        Args:
            outputs: Model outputs dictionary
            targets: Ground truth gaze [B, 2]
            inputs: Original input images [B, 1, H, W] (for cycle loss)

        Returns:
            Dictionary with loss components and total
        """
        gaze = outputs['gaze']
        gaze_init = outputs.get('gaze_init', gaze)
        correction = outputs.get('correction', torch.zeros_like(gaze))
        rendered = outputs.get('rendered', None)

        # Regression loss (MAE)
        loss_reg = F.l1_loss(gaze, targets)

        # Cycle-consistency loss
        if rendered is not None and inputs is not None:
            # Resize inputs to match rendered
            if inputs.shape != rendered.shape:
                inputs_resized = F.interpolate(
                    inputs, size=rendered.shape[2:],
                    mode='bilinear', align_corners=False
                )
            else:
                inputs_resized = inputs
            loss_cycle = F.l1_loss(rendered, inputs_resized)
        else:
            loss_cycle = torch.tensor(0.0, device=gaze.device)

        # ACM regularization (minimize correction magnitude)
        loss_acm = torch.mean(correction ** 2)

        # Total loss
        loss_total = (
            self.lambda_reg * loss_reg +
            self.lambda_cycle * loss_cycle +
            self.lambda_acm * loss_acm
        )

        # Calculate angular error in degrees for monitoring
        with torch.no_grad():
            gaze_deg = gaze.clone()
            gaze_deg[:, 0] *= self.yaw_max
            gaze_deg[:, 1] *= self.pitch_max

            targets_deg = targets.clone()
            targets_deg[:, 0] *= self.yaw_max
            targets_deg[:, 1] *= self.pitch_max

            angular_error = torch.mean(torch.abs(gaze_deg - targets_deg))

        return {
            'total': loss_total,
            'reg': loss_reg,
            'cycle': loss_cycle,
            'acm': loss_acm,
            'angular_error': angular_error
        }


# ============================================================================
# CELL 5: Training Pipeline
# ============================================================================

class PhysGazeTrainer:
    """
    Training pipeline for PhysGaze model.
    """

    def __init__(
        self,
        model: PhysGaze,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_reg: float = 1.0,
        lambda_cycle: float = 0.2,
        lambda_acm: float = 0.1,
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer (AdamW as per paper)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )

        # Loss function
        self.criterion = PhysGazeLoss(
            lambda_reg=lambda_reg,
            lambda_cycle=lambda_cycle,
            lambda_acm=lambda_acm
        )

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'outlier_rate': []
        }

        self.best_val_mae = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_mae = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images, return_all=True)

            # Calculate loss
            losses = self.criterion(outputs, targets, images)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += losses['total'].item()
            total_mae += losses['angular_error'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mae': f"{losses['angular_error'].item():.2f}°"
            })

        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        total_mae = 0
        num_batches = 0
        outliers = 0
        total_samples = 0

        for images, targets in tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]'):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images, return_all=False)

            # Calculate loss
            losses = self.criterion(outputs, targets)

            # Count outliers
            if hasattr(self.model, 'acm'):
                valid = self.model.acm.is_anatomically_valid(outputs['gaze'])
                outliers += (~valid).sum().item()

            total_loss += losses['total'].item()
            total_mae += losses['angular_error'].item()
            num_batches += 1
            total_samples += images.shape[0]

        outlier_rate = (outliers / total_samples) * 100 if total_samples > 0 else 0

        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'outlier_rate': outlier_rate
        }

    def train(self, num_epochs: int = 50, save_best: bool = True):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['outlier_rate'].append(val_metrics['outlier_rate'])

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('MAE/train', train_metrics['mae'], epoch)
            self.writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
            self.writer.add_scalar('Outlier_Rate', val_metrics['outlier_rate'], epoch)
            self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)

            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}°")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}°, Outliers: {val_metrics['outlier_rate']:.1f}%")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if save_best and val_metrics['mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['mae']
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ★ New best model saved! (MAE: {self.best_val_mae:.2f}°)")

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation MAE: {self.best_val_mae:.2f}°")
        print("=" * 60)

        # Save final model
        self.save_checkpoint(num_epochs - 1, 'final_model.pt')

        return self.history

    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'history': self.history
        }
        torch.save(checkpoint, self.log_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_mae = checkpoint['best_val_mae']
        self.history = checkpoint['history']
        return checkpoint['epoch']


# ============================================================================
# CELL 6: Evaluation and Visualization
# ============================================================================

class PhysGazeEvaluator:
    """
    Evaluation and visualization utilities for PhysGaze.
    """

    def __init__(self, model: PhysGaze, device: torch.device = device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.yaw_max = 55.0
        self.pitch_max = 40.0

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive evaluation on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        all_preds = []
        all_targets = []
        all_corrections = []

        for images, targets in tqdm(test_loader, desc='Evaluating'):
            images = images.to(self.device)
            outputs = self.model(images, return_all=False)

            all_preds.append(outputs['gaze'].cpu())
            all_targets.append(targets)
            all_corrections.append(outputs['correction'].cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        corrections = torch.cat(all_corrections, dim=0).numpy()

        # Convert to degrees
        preds_deg = preds.copy()
        preds_deg[:, 0] *= self.yaw_max
        preds_deg[:, 1] *= self.pitch_max

        targets_deg = targets.copy()
        targets_deg[:, 0] *= self.yaw_max
        targets_deg[:, 1] *= self.pitch_max

        # Calculate metrics
        errors = np.abs(preds_deg - targets_deg)
        angular_errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)

        # Outlier detection
        outlier_yaw = 60.0
        outlier_pitch = 45.0
        outliers = (np.abs(preds_deg[:, 0]) > outlier_yaw) | (np.abs(preds_deg[:, 1]) > outlier_pitch)

        # Extreme pose analysis (>40° head pose)
        extreme_mask = (np.abs(targets_deg[:, 0]) > 40) | (np.abs(targets_deg[:, 1]) > 30)

        metrics = {
            'mae': np.mean(angular_errors),
            'mae_yaw': np.mean(errors[:, 0]),
            'mae_pitch': np.mean(errors[:, 1]),
            'std': np.std(angular_errors),
            'median': np.median(angular_errors),
            'p95': np.percentile(angular_errors, 95),
            'outlier_rate': np.mean(outliers) * 100,
            'extreme_pose_mae': np.mean(angular_errors[extreme_mask]) if np.any(extreme_mask) else 0,
            'normal_pose_mae': np.mean(angular_errors[~extreme_mask]) if np.any(~extreme_mask) else 0,
            'mean_correction': np.mean(np.abs(corrections))
        }

        return metrics, preds_deg, targets_deg, angular_errors

    def plot_results(
        self,
        preds_deg: np.ndarray,
        targets_deg: np.ndarray,
        angular_errors: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Generate comprehensive visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Scatter plot of predictions vs targets
        ax = axes[0, 0]
        ax.scatter(targets_deg[:, 0], preds_deg[:, 0], alpha=0.3, s=5, c='#0ff4c6', label='Yaw')
        ax.scatter(targets_deg[:, 1], preds_deg[:, 1], alpha=0.3, s=5, c='#ff6b9d', label='Pitch')
        ax.plot([-60, 60], [-60, 60], 'k--', linewidth=1)
        ax.set_xlabel('Ground Truth (°)')
        ax.set_ylabel('Prediction (°)')
        ax.set_title('Predictions vs Ground Truth')
        ax.legend()
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.grid(True, alpha=0.3)

        # 2. Error distribution
        ax = axes[0, 1]
        ax.hist(angular_errors, bins=50, color='#a855f7', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(angular_errors), color='#0ff4c6', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(angular_errors):.2f}°')
        ax.axvline(np.median(angular_errors), color='#ff6b9d', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(angular_errors):.2f}°')
        ax.set_xlabel('Angular Error (°)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Gaze distribution with constraint zone
        ax = axes[0, 2]
        theta = np.linspace(0, 2*np.pi, 100)
        constraint_x = 55 * np.cos(theta)
        constraint_y = 40 * np.sin(theta)
        ax.fill(constraint_x, constraint_y, color='#0ff4c6', alpha=0.1, label='Valid Region')
        ax.plot(constraint_x, constraint_y, color='#0ff4c6', linestyle='--', linewidth=2)
        ax.scatter(preds_deg[:, 0], preds_deg[:, 1], alpha=0.3, s=5, c='#a855f7')
        ax.set_xlabel('Yaw (°)')
        ax.set_ylabel('Pitch (°)')
        ax.set_title('Gaze Distribution with Constraints')
        ax.set_xlim(-80, 80)
        ax.set_ylim(-60, 60)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 4. Error by gaze angle
        ax = axes[1, 0]
        gaze_magnitude = np.sqrt(targets_deg[:, 0]**2 + targets_deg[:, 1]**2)
        bins = [0, 10, 20, 30, 40, 50, 60]
        bin_errors = []
        bin_centers = []
        for i in range(len(bins) - 1):
            mask = (gaze_magnitude >= bins[i]) & (gaze_magnitude < bins[i+1])
            if np.any(mask):
                bin_errors.append(np.mean(angular_errors[mask]))
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        ax.bar(bin_centers, bin_errors, width=8, color='#0ff4c6', edgecolor='white')
        ax.set_xlabel('Gaze Magnitude (°)')
        ax.set_ylabel('Mean Error (°)')
        ax.set_title('Error vs Gaze Angle')
        ax.grid(True, alpha=0.3)

        # 5. Error heatmap
        ax = axes[1, 1]
        heatmap, xedges, yedges = np.histogram2d(
            targets_deg[:, 0], targets_deg[:, 1],
            bins=20, range=[[-50, 50], [-40, 30]],
            weights=angular_errors
        )
        counts, _, _ = np.histogram2d(
            targets_deg[:, 0], targets_deg[:, 1],
            bins=20, range=[[-50, 50], [-40, 30]]
        )
        heatmap = np.divide(heatmap, counts, where=counts > 0)
        heatmap[counts == 0] = np.nan
        im = ax.imshow(heatmap.T, origin='lower', extent=[-50, 50, -40, 30],
                       aspect='auto', cmap='magma')
        plt.colorbar(im, ax=ax, label='Mean Error (°)')
        ax.set_xlabel('Yaw (°)')
        ax.set_ylabel('Pitch (°)')
        ax.set_title('Error Heatmap')

        # 6. Cumulative error distribution
        ax = axes[1, 2]
        sorted_errors = np.sort(angular_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cumulative, color='#a855f7', linewidth=2)
        for thresh in [5, 10, 15]:
            pct = np.mean(angular_errors <= thresh) * 100
            ax.axvline(thresh, color='#64748b', linestyle=':', alpha=0.5)
            ax.text(thresh + 0.5, pct + 2, f'{pct:.1f}%', fontsize=9)
        ax.set_xlabel('Angular Error (°)')
        ax.set_ylabel('Cumulative %')
        ax.set_title('Cumulative Error Distribution')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def visualize_acm_effect(self, num_samples: int = 1000):
        """Visualize the ACM correction effect."""
        # Generate random gaze predictions (some invalid)
        raw_gaze = np.random.randn(num_samples, 2) * 1.5  # Some will exceed [-1, 1]

        raw_tensor = torch.tensor(raw_gaze, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            corrected, correction = self.model.acm(raw_tensor)
            corrected = corrected.cpu().numpy()
            correction = correction.cpu().numpy()

        # Convert to degrees
        raw_deg = raw_gaze.copy()
        raw_deg[:, 0] *= self.yaw_max
        raw_deg[:, 1] *= self.pitch_max

        corrected_deg = corrected.copy()
        corrected_deg[:, 0] *= self.yaw_max
        corrected_deg[:, 1] *= self.pitch_max

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Before ACM
        ax = axes[0]
        theta = np.linspace(0, 2*np.pi, 100)
        ax.fill(55*np.cos(theta), 40*np.sin(theta), color='#0ff4c6', alpha=0.1, label='Valid Region')
        ax.plot(55*np.cos(theta), 40*np.sin(theta), color='#0ff4c6', linestyle='--', linewidth=2)

        # Identify outliers
        invalid = (np.abs(raw_deg[:, 0]) > 60) | (np.abs(raw_deg[:, 1]) > 45)

        ax.scatter(raw_deg[~invalid, 0], raw_deg[~invalid, 1],
                   c='#22c55e', s=1.5, alpha=0.5, label='Valid')
        ax.scatter(raw_deg[invalid, 0], raw_deg[invalid, 1],
                   c='#ef4444', s=10, alpha=0.5, label='Invalid')

        ax.set_xlabel('Yaw (°)')
        ax.set_ylabel('Pitch (°)')
        ax.set_title(f'Before ACM (Outliers: {np.mean(invalid)*100:.1f}%)')
        ax.set_xlim(-80, 80)
        ax.set_ylim(-60, 60)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # After ACM
        ax = axes[1]
        ax.fill(55*np.cos(theta), 40*np.sin(theta), color='#0ff4c6', alpha=0.1, label='Valid Region')
        ax.plot(55*np.cos(theta), 40*np.sin(theta), color='#0ff4c6', linestyle='--', linewidth=2)

        # Check if any remain invalid after correction
        invalid_after = (np.abs(corrected_deg[:, 0]) > 60) | (np.abs(corrected_deg[:, 1]) > 45)

        ax.scatter(corrected_deg[~invalid_after, 0], corrected_deg[~invalid_after, 1],
                   c='#22c55e', s=1.5, alpha=0.5, label='Valid')
        ax.scatter(corrected_deg[invalid_after, 0], corrected_deg[invalid_after, 1],
                   c='#ef4444', s=10, alpha=0.5, label='Invalid')

        # Draw correction arrows for some samples
        for i in range(0, min(num_samples, 50), 10):
            if invalid[i]:
                ax.annotate('', xy=(corrected_deg[i, 0], corrected_deg[i, 1]),
                           xytext=(raw_deg[i, 0], raw_deg[i, 1]),
                           arrowprops=dict(arrowstyle='->', color='#ff6b9d', alpha=0.3))

        ax.set_xlabel('Yaw (°)')
        ax.set_ylabel('Pitch (°)')
        ax.set_title(f'After ACM (Outliers: {np.mean(invalid_after)*100:.1f}%)')
        ax.set_xlim(-80, 80)
        ax.set_ylim(-60, 60)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()


# ============================================================================
# CELL 7: Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("PhysGaze: Physics-Informed Gaze Estimation")
    print("=" * 60)

    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    # Option 1: Download real MPIIGaze dataset
    use_real_data = True  # Set to True if you want to download real data
    download_success = False

    if use_real_data:
        print("\n1. Downloading MPIIGaze dataset...")
        download_success = download_mpiigaze(root_dir='./data/MPIIGaze')
        
        # Check if dataset exists after download attempt
        dataset_path = Path('./data/MPIIGaze/Data')
        if dataset_path.exists():
            print(f"Dataset found at: {dataset_path}")
            download_success = True
        else:
            print("Dataset not found. Checking alternative locations...")
            # Try to find the dataset
            for path in Path('./data').rglob('*'):
                if 'MPIIGaze' in str(path):
                    print(f"Found MPIIGaze related file: {path}")
            
            print("\nUsing synthetic data instead.")
            use_real_data = False
            download_success = False

    # Create datasets
    print("\n2. Creating datasets...")
    
    # Disable debug mode to reduce console output
    debug_mode = False
    
    if use_real_data and download_success:
        # Real MPIIGaze dataset - try different possible locations
        possible_paths = [
            './data/MPIIGaze',
            './data/MPIIGaze/MPIIGaze',
            './data/MPIIGaze/Data'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"Found dataset at: {path}")
                root_dir = path
                break
        else:
            print("Could not find MPIIGaze dataset, using synthetic data")
            root_dir = None
            use_real_data = False
        
        if use_real_data:
            try:
                train_dataset = MPIIGazeDataset(
                    root_dir=root_dir, 
                    split='train',
                    debug=debug_mode
                )
                val_dataset = MPIIGazeDataset(
                    root_dir=root_dir, 
                    split='val',
                    debug=debug_mode
                )
                test_dataset = MPIIGazeDataset(
                    root_dir=root_dir, 
                    split='test',
                    debug=debug_mode
                )
            except Exception as e:
                print(f"Error loading real dataset: {e}")
                print("Falling back to synthetic data...")
                use_real_data = False
    
    if not use_real_data:
        # Synthetic dataset for testing
        print("Using synthetic dataset for demonstration.")
        train_dataset = SyntheticMPIIGaze(num_samples=8000, split='train')
        val_dataset = SyntheticMPIIGaze(num_samples=1000, split='val')
        test_dataset = SyntheticMPIIGaze(num_samples=1000, split='test')

    # Check dataset sizes before creating DataLoaders
    print(f"\nDataset sizes:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Ensure datasets have samples
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("\nWARNING: One or more datasets have zero samples!")
        print("Creating synthetic datasets instead...")
        
        train_dataset = SyntheticMPIIGaze(num_samples=8000, split='train')
        val_dataset = SyntheticMPIIGaze(num_samples=1000, split='val')
        test_dataset = SyntheticMPIIGaze(num_samples=1000, split='test')
        
        print(f"New sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Only create DataLoaders if datasets have samples
    if len(train_dataset) > 0 and len(val_dataset) > 0 and len(test_dataset) > 0:
        # Create data loaders with smaller batch size for debugging
        batch_size = 32  # Reduced for safety
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Set to 0 for debugging
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Set to 0 for debugging
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Set to 0 for debugging
            pin_memory=True
        )

        # Create model
        print("\n3. Creating PhysGaze model...")
        model = PhysGaze(
            pretrained_backbone=False,  # Set to False for faster initialization
            use_acm=True,
            use_renderer=True,
            image_size=(36, 60)
        )
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test a single forward pass
        print("\n4. Testing model with a single batch...")
        try:
            # Get a sample batch
            sample_batch = next(iter(train_loader))
            images, targets = sample_batch
            
            print(f"Batch - Images shape: {images.shape}, Targets shape: {targets.shape}")
            
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Test forward pass
            model = model.to(device)
            outputs = model(images, return_all=True)
            
            print("Forward pass successful!")
            print(f"Output gaze shape: {outputs['gaze'].shape}")
            
            if 'rendered' in outputs:
                print(f"Rendered images shape: {outputs['rendered'].shape}")
            
        except Exception as e:
            print(f"Error during test forward pass: {e}")
            import traceback
            traceback.print_exc()
            return

        # Create trainer
        print("\n5. Creating trainer...")
        trainer = PhysGazeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=1e-4,
            lambda_reg=1.0,
            lambda_cycle=0.2,
            lambda_acm=0.1,
            log_dir='./logs/physgaze'
        )

        # Train model for fewer epochs for testing
        print("\n6. Starting training...")
        try:
            history = trainer.train(num_epochs=5, save_best=True)  # Reduced epochs for testing
        except Exception as e:
            print(f"Error during training: {e}")
            return

        # Evaluate model
        print("\n7. Evaluating model...")
        try:
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

            # Visualize results
            print("\n8. Generating visualizations...")
            evaluator.plot_results(
                preds_deg, targets_deg, angular_errors,
                save_path='./logs/physgaze/results.png'
            )

            # Visualize ACM effect
            if model.use_acm:
                print("\n9. Visualizing ACM effect...")
                evaluator.visualize_acm_effect(num_samples=500)

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nERROR: Cannot proceed - one or more datasets have zero samples.")
        print("Please check your dataset path or use synthetic data.")

    print("\n" + "=" * 60)
    print("PhysGaze pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()