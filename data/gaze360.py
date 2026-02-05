"""
Gaze360 Dataset Loader
Reference: Kellnhofer et al., "Gaze360: Physically Unconstrained Gaze Estimation in the Wild" (ICCV 2019)
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import json
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Gaze360Dataset(Dataset):
    """
    Gaze360 Dataset for full-sphere gaze estimation.
    
    The dataset contains 172,000 images from 238 subjects with full 360-degree
    gaze annotations, captured in unconstrained environments.
    
    Reference: Kellnhofer et al., "Gaze360: Physically Unconstrained Gaze Estimation in the Wild" (ICCV 2019)
    """
    
    def __init__(
        self,
        root_dir: str = './data/Gaze360',
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        normalize_gaze: bool = True,
        load_images: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        debug: bool = False
    ):
        """
        Args:
            root_dir: Root directory of the Gaze360 dataset
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            normalize_gaze: Whether to normalize gaze vectors
            load_images: Whether to load images or just metadata
            image_size: Size to resize images to
            debug: Enable debug print statements
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.normalize_gaze = normalize_gaze
        self.load_images = load_images
        self.image_size = image_size
        self.debug = debug
        
        # Full sphere gaze range
        self.yaw_range = [-180, 180]  # degrees
        self.pitch_range = [-90, 90]   # degrees
        
        # Dataset metadata
        self.samples = []
        self._load_dataset()
        
        if debug:
            print(f"\nGaze360 Dataset loaded: {split}")
            print(f"Number of samples: {len(self.samples)}")
            if len(self.samples) > 0:
                print(f"Sample keys: {list(self.samples[0].keys())}")
    
    def _load_dataset(self):
        """Load Gaze360 dataset from structure."""
        # Check if dataset exists
        if not self.root_dir.exists():
            print(f"Warning: Gaze360 directory not found: {self.root_dir}")
            print("Please download the dataset from: http://gaze360.csail.mit.edu")
            return
        
        # Load split information
        split_file = self.root_dir / 'splits' / f'{self.split}.txt'
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
        else:
            # If split file doesn't exist, scan for images
            print(f"Split file not found: {split_file}")
            print("Scanning for images...")
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend([str(p.relative_to(self.root_dir)) 
                                  for p in self.root_dir.rglob(ext)])
            # Take subset for train/val/test
            if self.split == 'train':
                image_paths = image_paths[:int(0.7 * len(image_paths))]
            elif self.split == 'val':
                image_paths = image_paths[int(0.7 * len(image_paths)):int(0.85 * len(image_paths))]
            else:  # test
                image_paths = image_paths[int(0.85 * len(image_paths)):]
        
        # Load labels
        labels_file = self.root_dir / 'labels.txt'
        labels_dict = {}
        
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            img_name = parts[0]
                            # Gaze vector (x, y, z)
                            gaze = list(map(float, parts[1:4]))
                            labels_dict[img_name] = gaze
        
        # Create samples
        for img_rel_path in image_paths:
            img_path = self.root_dir / img_rel_path
            
            if not img_path.exists():
                if self.debug:
                    print(f"Image not found: {img_path}")
                continue
            
            # Get gaze label
            gaze_vector = None
            img_name = img_path.name
            
            if img_name in labels_dict:
                gaze_vector = np.array(labels_dict[img_name], dtype=np.float32)
            else:
                # Try to extract from parent directory name or file name
                # Gaze360 sometimes encodes gaze in directory structure
                try:
                    # Example: path like 'subject_001/yaw_30_pitch_-10/image.jpg'
                    parent_dir = img_path.parent.name
                    if 'yaw' in parent_dir and 'pitch' in parent_dir:
                        import re
                        yaw_match = re.search(r'yaw_([-\d]+)', parent_dir)
                        pitch_match = re.search(r'pitch_([-\d]+)', parent_dir)
                        if yaw_match and pitch_match:
                            yaw = float(yaw_match.group(1))
                            pitch = float(pitch_match.group(1))
                            # Convert spherical to cartesian
                            gaze_vector = self._spherical_to_cartesian(yaw, pitch)
                except:
                    pass
            
            # If still no gaze, skip or use default
            if gaze_vector is None:
                if self.debug:
                    print(f"No gaze label for: {img_name}")
                # Use forward gaze as default
                gaze_vector = np.array([0, 0, -1], dtype=np.float32)
            
            # Get face bounding box (if available)
            bbox_file = img_path.with_suffix('.bbox')
            face_bbox = None
            
            if bbox_file.exists():
                try:
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                        if 'face' in bbox_data:
                            face_bbox = bbox_data['face']
                except:
                    pass
            
            # Get head pose (if available)
            pose_file = img_path.with_suffix('.pose')
            head_pose = None
            
            if pose_file.exists():
                try:
                    with open(pose_file, 'r') as f:
                        pose_data = json.load(f)
                        if 'head_pose' in pose_data:
                            head_pose = pose_data['head_pose']
                except:
                    pass
            
            sample = {
                'image_path': str(img_path),
                'gaze_vector': gaze_vector,
                'face_bbox': face_bbox,
                'head_pose': head_pose,
                'subject_id': self._extract_subject_id(str(img_rel_path))
            }
            
            self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")
    
    def _extract_subject_id(self, path: str) -> int:
        """Extract subject ID from path."""
        # Gaze360 paths often contain subject IDs
        import re
        match = re.search(r'subject[_\s]*(\d+)', path, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Try other patterns
        match = re.search(r'p(\d+)', path, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Use hash as fallback
        return hash(path) % 1000
    
    def _spherical_to_cartesian(self, yaw: float, pitch: float) -> np.ndarray:
        """Convert spherical coordinates (yaw, pitch) to cartesian gaze vector."""
        # Convert degrees to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Spherical to cartesian conversion
        x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        y = np.sin(pitch_rad)
        z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def _cartesian_to_spherical(self, gaze_vector: np.ndarray) -> Tuple[float, float]:
        """Convert cartesian gaze vector to spherical coordinates (yaw, pitch)."""
        x, y, z = gaze_vector
        
        # Normalize
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 0:
            x, y, z = x/norm, y/norm, z/norm
        
        # Calculate yaw and pitch
        yaw = np.degrees(np.arctan2(-x, -z))  # Negative signs for consistency
        pitch = np.degrees(np.arcsin(y))
        
        return yaw, pitch
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Crop face if bounding box available (will be handled in __getitem__)
            # For now, just resize
            if self.image_size:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            
            # Convert to float and normalize
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        if self.load_images:
            image = self._load_and_preprocess_image(sample['image_path'])
            
            # Convert to tensor and adjust dimensions
            image_tensor = torch.from_numpy(image).float()
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            # Convert to grayscale if needed
            if image_tensor.shape[0] == 3:
                # Simple grayscale conversion: weighted average
                image_tensor = 0.299 * image_tensor[0:1] + 0.587 * image_tensor[1:2] + 0.114 * image_tensor[2:3]
        else:
            # Return dummy image
            image_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        
        # Get gaze vector
        gaze_vector = sample['gaze_vector'].copy()
        
        # Convert to spherical coordinates
        yaw, pitch = self._cartesian_to_spherical(gaze_vector)
        
        # Normalize to [-1, 1] if requested
        if self.normalize_gaze:
            yaw_norm = yaw / 180.0  # Normalize by 180 degrees
            pitch_norm = pitch / 90.0  # Normalize by 90 degrees
            
            # Clip to valid range
            yaw_norm = np.clip(yaw_norm, -1.0, 1.0)
            pitch_norm = np.clip(pitch_norm, -1.0, 1.0)
            
            gaze_tensor = torch.tensor([yaw_norm, pitch_norm], dtype=torch.float32)
        else:
            gaze_tensor = torch.tensor([yaw, pitch], dtype=torch.float32)
        
        # Apply transforms if specified
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Additional metadata
        metadata = {
            'image_path': sample['image_path'],
            'subject_id': sample['subject_id'],
            'gaze_vector': torch.tensor(gaze_vector, dtype=torch.float32),
            'face_bbox': sample['face_bbox'],
            'head_pose': sample['head_pose']
        }
        
        return image_tensor, gaze_tensor, metadata


class SyntheticGaze360(Dataset):
    """
    Synthetic Gaze360-like dataset for testing without downloading real data.
    Generates full-sphere gaze samples with realistic variations.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        image_size: Tuple[int, int] = (224, 224),
        split: str = 'train',
        normalize_gaze: bool = True,
        include_head_pose: bool = False
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split
        self.normalize_gaze = normalize_gaze
        self.include_head_pose = include_head_pose
        
        # Full sphere ranges
        self.yaw_range = [-180, 180]
        self.pitch_range = [-90, 90]
        
        # Generate samples
        self.samples = self._generate_samples()
        
        print(f"Synthetic Gaze360 Dataset: {split}")
        print(f"Number of samples: {len(self.samples)}")
        print(f"Yaw range: {self.yaw_range[0]}° to {self.yaw_range[1]}°")
        print(f"Pitch range: {self.pitch_range[0]}° to {self.pitch_range[1]}°")
    
    def _generate_samples(self) -> List[Dict]:
        """Generate synthetic gaze samples."""
        samples = []
        
        for i in range(self.num_samples):
            # Sample gaze direction (full sphere)
            if self.split == 'train':
                # Training: more uniform distribution
                yaw = np.random.uniform(*self.yaw_range)
                pitch = np.random.uniform(*self.pitch_range)
            else:
                # Validation/Test: more challenging samples
                # Include more extreme angles
                if np.random.random() < 0.3:  # 30% extreme samples
                    yaw = np.random.choice([-150, -120, 120, 150]) + np.random.uniform(-10, 10)
                    pitch = np.random.choice([-70, -50, 50, 70]) + np.random.uniform(-10, 10)
                else:
                    yaw = np.random.uniform(-90, 90)
                    pitch = np.random.uniform(-45, 45)
            
            # Convert to cartesian vector
            gaze_vector = self._spherical_to_cartesian(yaw, pitch)
            
            # Generate synthetic head pose (if requested)
            head_pose = None
            if self.include_head_pose:
                # Small head rotations
                head_yaw = np.random.uniform(-30, 30)
                head_pitch = np.random.uniform(-20, 20)
                head_roll = np.random.uniform(-10, 10)
                head_pose = [head_yaw, head_pitch, head_roll]
            
            # Generate synthetic face bounding box
            h, w = self.image_size
            face_size = min(h, w) // 3
            x_min = np.random.randint(0, w - face_size)
            y_min = np.random.randint(0, h - face_size)
            face_bbox = [x_min, y_min, x_min + face_size, y_min + face_size]
            
            sample = {
                'gaze_vector': gaze_vector,
                'yaw': yaw,
                'pitch': pitch,
                'head_pose': head_pose,
                'face_bbox': face_bbox,
                'subject_id': np.random.randint(0, 100),
                'image_id': i
            }
            
            samples.append(sample)
        
        return samples
    
    def _spherical_to_cartesian(self, yaw: float, pitch: float) -> np.ndarray:
        """Convert spherical to cartesian coordinates."""
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        y = np.sin(pitch_rad)
        z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def _generate_synthetic_image(self, gaze_vector: np.ndarray) -> np.ndarray:
        """Generate a synthetic eye/face image based on gaze direction."""
        h, w = self.image_size
        image = np.zeros((h, w, 3), dtype=np.float32)
        
        # Extract yaw and pitch for eye positioning
        x, y, z = gaze_vector
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 0:
            x, y, z = x/norm, y/norm, z/norm
        
        yaw = np.degrees(np.arctan2(-x, -z))
        pitch = np.degrees(np.arcsin(y))
        
        # Center of face
        cx, cy = w // 2, h // 2
        
        # Draw face oval
        face_width = w // 2
        face_height = h // 2
        
        # Create meshgrid
        yy, xx = np.ogrid[:h, :w]
        
        # Face mask (ellipse)
        face_mask = ((xx - cx) / face_width) ** 2 + ((yy - cy) / face_height) ** 2 <= 1
        
        # Face color (skin tone)
        skin_color = np.array([0.9, 0.7, 0.6]) + np.random.uniform(-0.1, 0.1, 3)
        image[face_mask] = skin_color
        
        # Eye positions
        eye_y = cy - h // 8
        eye_spacing = w // 4
        left_eye_x = cx - eye_spacing // 2
        right_eye_x = cx + eye_spacing // 2
        
        # Eye parameters
        eye_width = w // 8
        eye_height = h // 10
        
        # Draw eyes
        for eye_x in [left_eye_x, right_eye_x]:
            # Eye socket (ellipse)
            eye_mask = ((xx - eye_x) / eye_width) ** 2 + ((yy - eye_y) / eye_height) ** 2 <= 1
            eye_mask = eye_mask & face_mask  # Only on face
            
            # Eye white
            image[eye_mask] = np.array([0.95, 0.95, 0.95]) + np.random.uniform(-0.05, 0.05, 3)
            
            # Iris position based on gaze
            iris_offset_x = (yaw / 180.0) * (eye_width // 2)
            iris_offset_y = (pitch / 90.0) * (eye_height // 2)
            iris_x = eye_x + iris_offset_x
            iris_y = eye_y + iris_offset_y
            
            # Iris (circle)
            iris_radius = eye_height // 2
            iris_mask = (xx - iris_x) ** 2 + (yy - iris_y) ** 2 <= iris_radius ** 2
            iris_mask = iris_mask & eye_mask
            
            # Iris color
            iris_color = np.array([0.4, 0.6, 0.2]) + np.random.uniform(-0.1, 0.1, 3)
            image[iris_mask] = iris_color
            
            # Pupil (smaller circle)
            pupil_radius = iris_radius // 2
            pupil_mask = (xx - iris_x) ** 2 + (yy - iris_y) ** 2 <= pupil_radius ** 2
            pupil_mask = pupil_mask & iris_mask
            image[pupil_mask] = np.array([0.1, 0.1, 0.1])
            
            # Specular highlight
            highlight_x = iris_x + 2
            highlight_y = iris_y - 2
            highlight_radius = 2
            highlight_mask = (xx - highlight_x) ** 2 + (yy - highlight_y) ** 2 <= highlight_radius ** 2
            highlight_mask = highlight_mask & iris_mask
            image[highlight_mask] = np.array([1.0, 1.0, 1.0])
        
        # Add some noise and texture
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        # Add some lighting variation
        light_gradient = np.linspace(0.9, 1.1, h).reshape(-1, 1, 1)
        image *= light_gradient
        
        return np.clip(image, 0, 1)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # Generate synthetic image
        image = self._generate_synthetic_image(sample['gaze_vector'])
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        # Convert to grayscale (single channel)
        image_tensor = 0.299 * image_tensor[0:1] + 0.587 * image_tensor[1:2] + 0.114 * image_tensor[2:3]
        
        # Get gaze
        yaw, pitch = sample['yaw'], sample['pitch']
        
        if self.normalize_gaze:
            yaw_norm = yaw / 180.0
            pitch_norm = pitch / 90.0
            gaze_tensor = torch.tensor([yaw_norm, pitch_norm], dtype=torch.float32)
        else:
            gaze_tensor = torch.tensor([yaw, pitch], dtype=torch.float32)
        
        # Metadata
        metadata = {
            'gaze_vector': torch.tensor(sample['gaze_vector'], dtype=torch.float32),
            'head_pose': torch.tensor(sample['head_pose'], dtype=torch.float32) if sample['head_pose'] else None,
            'face_bbox': sample['face_bbox'],
            'subject_id': sample['subject_id'],
            'image_id': sample['image_id']
        }
        
        return image_tensor, gaze_tensor, metadata


def download_gaze360(root_dir: str = './data/Gaze360') -> bool:
    """
    Download and extract the Gaze360 dataset.
    
    Note: Gaze360 is large (~100GB). This function provides instructions
    and downloads a smaller sample if available.
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Gaze360 Dataset Download")
    print("=" * 60)
    print("Note: Full Gaze360 dataset is ~100GB")
    print("Official download page: http://gaze360.csail.mit.edu")
    print("\nOptions:")
    print("1. Download full dataset (requires registration)")
    print("2. Download sample dataset (for testing)")
    print("=" * 60)
    
    # Check if dataset already exists
    if (root_path / 'images').exists() and len(list((root_path / 'images').glob('*.jpg'))) > 100:
        print("Gaze360 dataset already exists.")
        return True
    
    # Try to download a sample
    print("\nDownloading sample dataset...")
    
    # Sample dataset URL (if available)
    sample_url = "https://github.com/erkil/gaze-360-sample/releases/download/v1.0/gaze360_sample.zip"
    
    try:
        import urllib.request
        import zipfile
        
        sample_zip = root_path / 'gaze360_sample.zip'
        
        print(f"Downloading sample from: {sample_url}")
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}%", end='')
        
        urllib.request.urlretrieve(sample_url, sample_zip, reporthook=report_progress)
        print("\nDownload complete!")
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(sample_zip, 'r') as zip_ref:
            zip_ref.extractall(root_path)
        
        # Clean up
        os.remove(sample_zip)
        print("Extraction complete!")
        
        # Create necessary structure
        (root_path / 'splits').mkdir(exist_ok=True)
        
        # Create sample split files
        images = list(root_path.rglob('*.jpg')) + list(root_path.rglob('*.png'))
        if images:
            # Split into train/val/test
            n = len(images)
            train_split = images[:int(0.7 * n)]
            val_split = images[int(0.7 * n):int(0.85 * n)]
            test_split = images[int(0.85 * n):]
            
            # Write split files
            with open(root_path / 'splits' / 'train.txt', 'w') as f:
                for img in train_split:
                    f.write(str(img.relative_to(root_path)) + '\n')
            
            with open(root_path / 'splits' / 'val.txt', 'w') as f:
                for img in val_split:
                    f.write(str(img.relative_to(root_path)) + '\n')
            
            with open(root_path / 'splits' / 'test.txt', 'w') as f:
                for img in test_split:
                    f.write(str(img.relative_to(root_path)) + '\n')
            
            print(f"Created splits: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test")
        
        return True
        
    except Exception as e:
        print(f"\nError downloading sample: {e}")
        print("\nManual download instructions:")
        print("1. Visit: http://gaze360.csail.mit.edu")
        print("2. Request access to the dataset")
        print("3. Download and extract to: {}".format(root_path.absolute()))
        print("4. Ensure structure: Gaze360/images/...")
        return False


def create_gaze360_dataloader(
    root_dir: str = './data/Gaze360',
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_synthetic: bool = False,
    shuffle: bool = True,
    normalize_gaze: bool = True
) -> DataLoader:
    """
    Create a DataLoader for Gaze360 dataset.
    
    Args:
        root_dir: Dataset root directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size (H, W)
        use_synthetic: Use synthetic data if real data not available
        shuffle: Whether to shuffle the data
        normalize_gaze: Normalize gaze to [-1, 1]
    
    Returns:
        DataLoader instance
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale
    ])
    
    # Create dataset
    if use_synthetic or not Path(root_dir).exists():
        print(f"Using synthetic Gaze360 dataset for {split}")
        dataset = SyntheticGaze360(
            num_samples=1000 if split == 'train' else 200,
            image_size=image_size,
            split=split,
            normalize_gaze=normalize_gaze
        )
    else:
        dataset = Gaze360Dataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            normalize_gaze=normalize_gaze,
            image_size=image_size,
            load_images=True
        )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


# Test function
def test_gaze360_dataset():
    """Test the Gaze360 dataset loader."""
    print("Testing Gaze360 Dataset...")
    
    # Try to load real dataset
    try:
        dataset = Gaze360Dataset(
            root_dir='./data/Gaze360',
            split='train',
            normalize_gaze=True,
            load_images=False,  # Don't load images for quick test
            debug=True
        )
        
        print(f"Real dataset samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get one sample
            image, gaze, metadata = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Gaze tensor: {gaze}")
            print(f"Gaze (denormalized): Yaw={gaze[0].item()*180:.1f}°, Pitch={gaze[1].item()*90:.1f}°")
            print(f"Metadata keys: {list(metadata.keys())}")
    
    except Exception as e:
        print(f"Could not load real dataset: {e}")
        print("Testing with synthetic dataset...")
    
    # Test synthetic dataset
    synthetic = SyntheticGaze360(
        num_samples=10,
        image_size=(224, 224),
        split='train',
        normalize_gaze=True
    )
    
    print(f"\nSynthetic dataset samples: {len(synthetic)}")
    
    # Get one sample
    image, gaze, metadata = synthetic[0]
    print(f"Image shape: {image.shape}")
    print(f"Gaze tensor: {gaze}")
    print(f"Gaze (denormalized): Yaw={metadata['gaze_vector'][0].item()}, Pitch={metadata['gaze_vector'][1].item()}")
    print(f"Subject ID: {metadata['subject_id']}")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(10, len(synthetic))):
        img, gaze, _ = synthetic[i]
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i].set_title(f"Y:{gaze[0]:.2f}, P:{gaze[1]:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nGaze360 dataset test completed!")


if __name__ == '__main__':
    # Run test
    test_gaze360_dataset()