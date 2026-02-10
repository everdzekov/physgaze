import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Optional
from .preprocessing import normalize_image, augment_data

class GazeDataset(Dataset):
    """Base gaze dataset class"""
    
    def __init__(self, df: pd.DataFrame, transform: Optional[callable] = None, 
                 augment: bool = False):
        self.df = df
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image
        img = row['image']
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.expand_dims(img, 0)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img.transpose(2, 0, 1)
        
        # Get gaze angles
        yaw = float(row['gaze_yaw'])
        pitch = float(row['gaze_pitch'])
        
        # Apply augmentation
        if self.augment:
            img, gaze = augment_data(img, np.array([yaw, pitch]))
            yaw, pitch = gaze[0], gaze[1]
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        
        # Convert to tensor
        if not isinstance(img, torch.Tensor):
            img_tensor = torch.from_numpy(img).float()
        else:
            img_tensor = img
        
        gaze_tensor = torch.tensor([yaw, pitch], dtype=torch.float32)
        
        return img_tensor, gaze_tensor


class MPIIGazeLOSODataset(GazeDataset):
    """Dataset for MPIIGaze LOSO cross-validation"""
    
    def __init__(self, df: pd.DataFrame, test_subject: str, 
                 val_subjects: Optional[List[str]] = None, 
                 mode: str = 'train', **kwargs):
        """
        Args:
            df: DataFrame with all data
            test_subject: Subject held out for testing
            val_subjects: Subjects used for validation
            mode: 'train', 'val', or 'test'
        """
        self.test_subject = test_subject
        self.val_subjects = val_subjects if val_subjects else []
        self.mode = mode
        
        # Apply split
        df = self._apply_split(df)
        
        print(f"Created {mode} dataset: {len(df)} samples "
              f"(test subject: {test_subject})")
        
        super().__init__(df, **kwargs)
    
    def _apply_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply LOSO split"""
        df = df.copy()
        
        if self.mode == 'test':
            # Test set: only test subject
            df = df[df['subject'] == self.test_subject].copy()
        elif self.mode == 'val':
            # Validation set: specified validation subjects
            df = df[df['subject'].isin(self.val_subjects)].copy()
        else:
            # Training set: all subjects except test and validation
            exclude = [self.test_subject] + self.val_subjects
            df = df[~df['subject'].isin(exclude)].copy()
        
        return df


class Gaze360Dataset(GazeDataset):
    """Dataset for Gaze360"""
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, **kwargs)
