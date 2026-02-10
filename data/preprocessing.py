import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional

def normalize_image(image: np.ndarray, target_size: Tuple[int, int] = (36, 60)) -> np.ndarray:
    """Normalize eye image"""
    if image.ndim == 3:
        # Convert to grayscale
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 1:
            image = image.squeeze(2)
    
    # Resize if needed
    if image.shape != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension
    if image.ndim == 2:
        image = np.expand_dims(image, 0)
    
    return image

def augment_data(image: np.ndarray, gaze: np.ndarray, 
                 augment_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation"""
    if np.random.random() < augment_prob:
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
        
        # Random contrast
        contrast = np.random.uniform(0.8, 1.2)
        image = np.clip(0.5 + contrast * (image - 0.5), 0, 1)
        
        # Add Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)
    
    return image, gaze

def gaze_angles_to_vector(yaw: float, pitch: float) -> np.ndarray:
    """Convert yaw and pitch angles to 3D gaze vector"""
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    
    x = -np.sin(yaw_rad) * np.cos(pitch_rad)
    y = -np.sin(pitch_rad)
    z = -np.cos(yaw_rad) * np.cos(pitch_rad)
    
    vector = np.array([x, y, z])
    norm = np.linalg.norm(vector)
    
    if norm > 0:
        vector = vector / norm
    
    return vector

def clip_gaze_angles(yaw: float, pitch: float, 
                     yaw_limit: float = 50.0, 
                     pitch_limit: float = 35.0) -> Tuple[float, float]:
    """Clip gaze angles to physiological limits"""
    yaw = np.clip(yaw, -yaw_limit, yaw_limit)
    pitch = np.clip(pitch, -pitch_limit, pitch_limit)
    return yaw, pitch
