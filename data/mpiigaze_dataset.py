import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .mpiigaze_loader import load_all_mpiigaze_subjects_fixed

class MPIIGazeDataset(Dataset):
    """Dataset class for MPIIGaze with gaze angle filtering."""
    
    def __init__(self, split='train', test_subject='p00', samples_per_subject=200, 
                 augment=True, debug=False, filter_extreme_gaze=True):
        self.split = split
        self.test_subject = test_subject
        self.augment = augment and (split == 'train')
        self.debug = debug
        self.filter_extreme_gaze = filter_extreme_gaze
        
        print(f"\n📁 Creating {split.upper()} dataset...")
        print(f"  Test subject: {test_subject}")
        print(f"  Filter extreme gaze: {filter_extreme_gaze}")
        
        # Load all data
        self.data = load_all_mpiigaze_subjects_fixed(
            base_dir='./data/MPIIGaze/MPIIGaze',
            samples_per_subject=samples_per_subject,
            debug=debug
        )
        
        if len(self.data) == 0:
            print("⚠️ No data loaded. Please check the data path.")
            # Create empty DataFrame with correct columns
            self.data = pd.DataFrame(columns=['subject', 'image_data', 'gaze_x', 'gaze_y'])
        else:
            # Filter extreme gaze angles if requested
            if self.filter_extreme_gaze:
                self._filter_extreme_gaze()
            
            # Apply split
            self._apply_split()
        
        print(f"\n✅ {split.upper()} DATASET READY:")
        print(f"  Samples: {len(self.data):,}")
        if len(self.data) > 0:
            print(f"  Subjects: {self.data['subject'].nunique()}")
            print(f"  Gaze X (yaw) range: [{self.data['gaze_x'].min():.1f}°, {self.data['gaze_x'].max():.1f}°]")
            print(f"  Gaze Y (pitch) range: [{self.data['gaze_y'].min():.1f}°, {self.data['gaze_y'].max():.1f}°]")
    
    def _filter_extreme_gaze(self):
        """Filter out samples with extreme gaze angles."""
        initial_count = len(self.data)
        
        # Filter for reasonable gaze angles
        mask = (self.data['gaze_x'].abs() <= 45) & (self.data['gaze_y'].abs() <= 30)
        self.data = self.data[mask].copy()
        
        filtered_count = initial_count - len(self.data)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} samples with extreme gaze angles")
            print(f"  Remaining: {len(self.data)} samples")
    
    def _apply_split(self):
        """Apply train/val/test split based on subject."""
        if self.split == 'test':
            # Test on specified subject
            if self.test_subject in self.data['subject'].unique():
                self.data = self.data[self.data['subject'] == self.test_subject].copy()
                print(f"  Using test subject: {self.test_subject} ({len(self.data)} samples)")
            else:
                print(f"  ⚠️ Test subject {self.test_subject} not found. Using 10% random samples.")
                self.data = self.data.sample(frac=0.1, random_state=42)
        
        elif self.split == 'val':
            # Validation: 10% of non-test data
            non_test = self.data[self.data['subject'] != self.test_subject]
            if len(non_test) > 0:
                self.data = non_test.sample(frac=0.1, random_state=42)
                print(f"  Validation set: {len(self.data)} samples from non-test subjects")
            else:
                print(f"  ⚠️ No non-test subjects found for validation.")
                self.data = pd.DataFrame()
        
        else:  # train
            # Train: all data except test subject
            self.data = self.data[self.data['subject'] != self.test_subject].copy()
            print(f"  Training set: {len(self.data)} samples (excluding {self.test_subject})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if len(self.data) == 0:
            # Return dummy data if no real data
            dummy_img = np.random.rand(36, 60).astype(np.float32)
            dummy_gaze = np.array([0.0, 0.0], dtype=np.float32)
            return torch.from_numpy(dummy_img).unsqueeze(0), torch.from_numpy(dummy_gaze)
        
        row = self.data.iloc[idx]
        
        # Get image
        image = row['image_data'].copy()
        
        # Ensure 2D
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = np.mean(image, axis=2)
        
        # Get gaze
        gaze_x, gaze_y = row['gaze_x'], row['gaze_y']
        
        # Apply augmentations if training
        if self.augment:
            # Brightness/contrast augmentation
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.9, 1.1)
            image = np.clip((image - 0.5) * contrast + 0.5 + (brightness - 1.0), 0, 1)
            
            # Horizontal flip with 50% probability
            if np.random.random() < 0.5:
                image = np.fliplr(image).copy()
                gaze_x = -gaze_x  # Flip horizontal gaze
        
        # Convert to tensor [1, 36, 60]
        image_tensor = torch.from_numpy(image).float()
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Normalize gaze to [-1, 1] range
        # MPIIGaze typical ranges: yaw ±30°, pitch ±20°
        gaze_tensor = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)
        gaze_tensor[0] = gaze_tensor[0] / 30.0  # Normalize yaw
        gaze_tensor[1] = gaze_tensor[1] / 20.0  # Normalize pitch
        gaze_tensor = torch.clamp(gaze_tensor, -1, 1)
        
        return image_tensor, gaze_tensor
