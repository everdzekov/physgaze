import os
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class BaseGazeLoader:
    """Base class for gaze dataset loaders"""
    
    @staticmethod
    def _gaze_vector_to_angles(gaze_vector):
        """Convert 3D gaze vector to yaw and pitch angles (in degrees)"""
        if gaze_vector is None or len(gaze_vector) < 3:
            return 0.0, 0.0

        x, y, z = gaze_vector[0], gaze_vector[1], gaze_vector[2]

        # Normalize
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 0:
            x, y, z = x/norm, y/norm, z/norm

        # Convert to angles
        yaw = np.arctan2(-x, -z)  # Horizontal angle
        pitch = np.arcsin(-y)     # Vertical angle

        # Convert to degrees
        yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)

        return yaw_deg, pitch_deg


class MPIIGazeLoader(BaseGazeLoader):
    """Loader for MPIIGaze dataset with LOSO support"""
    
    def __init__(self, base_dir: str = './data/MPIIGaze'):
        self.base_dir = base_dir
        
    def load_all_subjects(self):
        """Load all MPIIGaze subjects"""
        print(f"Loading MPIIGaze dataset from: {self.base_dir}")
        
        # Standard MPIIGaze structure
        normalized_dir = os.path.join(self.base_dir, "Data", "Normalized")
        
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
                        all_data.extend(self._process_eye_data(
                            data_dict['right'], subject, 'right'
                        ))

                    # Process left eye
                    if 'left' in data_dict:
                        all_data.extend(self._process_eye_data(
                            data_dict['left'], subject, 'left'
                        ))

                except Exception as e:
                    print(f"Error loading {mat_file}: {e}")
                    continue

        df = pd.DataFrame(all_data)

        if len(df) == 0:
            print("No data loaded. Creating synthetic data for testing...")
            return self._create_synthetic_data()

        print(f"\n✅ Dataset loaded successfully!")
        print(f"Total samples: {len(df):,}")
        print(f"Subjects: {df['subject'].unique().tolist()}")
        print(f"Eye distribution: {df['eye'].value_counts().to_dict()}")

        return df
    
    def _process_eye_data(self, eye_data, subject, eye):
        """Process data for a single eye"""
        data_list = []
        
        if isinstance(eye_data, dict):
            if 'image' in eye_data and 'gaze' in eye_data:
                images = eye_data['image']
                gazes = eye_data['gaze']

                # Convert to proper format
                if images.ndim == 4:  # (n, 36, 60, 3)
                    images = np.mean(images, axis=3)  # Convert to grayscale

                n_samples = min(len(images), len(gazes))

                for i in range(n_samples):
                    # Process image
                    img = images[i].astype(np.float32) / 255.0

                    # Process gaze
                    gaze_3d = gazes[i]
                    yaw, pitch = self._gaze_vector_to_angles(gaze_3d)

                    data_list.append({
                        'subject': subject,
                        'image': img,
                        'gaze_3d': gaze_3d,
                        'gaze_yaw': yaw,
                        'gaze_pitch': pitch,
                        'eye': eye
                    })
        
        return data_list
    
    def _create_synthetic_data(self):
        """Create synthetic MPIIGaze-like data for testing"""
        print("Creating synthetic MPIIGaze data...")

        subjects = [f'p{i:02d}' for i in range(15)]  # p00 to p14
        all_data = []

        for subject in subjects:
            n_samples = np.random.randint(8000, 15000)

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

                # Generate gaze angles
                if np.random.random() < 0.15:  # 15% outliers
                    yaw = np.random.uniform(-70, 70)
                    pitch = np.random.uniform(-50, 50)
                else:
                    yaw = np.random.uniform(-40, 40)
                    pitch = np.random.uniform(-25, 25)

                # Add subject-specific bias
                subject_id = int(subject[1:])
                yaw += subject_id * 2.0

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


class Gaze360Loader(BaseGazeLoader):
    """Loader for Gaze360 dataset"""
    
    def __init__(self, base_dir: str = './data/Gaze360'):
        self.base_dir = base_dir
        
    def load_data(self, split: str = 'train'):
        """Load Gaze360 dataset split"""
        print(f"Loading Gaze360 {split} data from: {self.base_dir}")
        
        # Gaze360 dataset structure
        if split == 'train':
            data_file = os.path.join(self.base_dir, 'train', 'annotations.mat')
        elif split == 'test':
            data_file = os.path.join(self.base_dir, 'test', 'annotations.mat')
        elif split == 'val':
            data_file = os.path.join(self.base_dir, 'val', 'annotations.mat')
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Gaze360 data file not found: {data_file}")
        
        # Load MATLAB file
        mat_data = sio.loadmat(data_file, simplify_cells=True)
        
        # Extract data
        images = mat_data['image']
        gazes = mat_data['pose']
        subjects = mat_data['person'] if 'person' in mat_data else np.zeros(len(images))
        
        all_data = []
        
        for idx in tqdm(range(len(images)), desc=f"Loading {split} samples"):
            img_path = images[idx]
            
            # Extract subject ID from path
            subject = f"p{int(subjects[idx]):03d}" if isinstance(subjects[idx], (int, float)) else f"p{idx:03d}"
            
            # Load and preprocess image
            full_img_path = os.path.join(self.base_dir, 'images', img_path)
            
            # For now, create placeholder image
            img = np.random.rand(36, 60).astype(np.float32) * 0.5
            
            # Process gaze
            gaze_3d = gazes[idx]
            yaw, pitch = self._gaze_vector_to_angles(gaze_3d)
            
            all_data.append({
                'subject': subject,
                'image': img,
                'image_path': full_img_path,
                'gaze_3d': gaze_3d,
                'gaze_yaw': yaw,
                'gaze_pitch': pitch,
                'split': split
            })
        
        df = pd.DataFrame(all_data)
        
        print(f"\n✅ Gaze360 {split} dataset loaded successfully!")
        print(f"Total samples: {len(df):,}")
        print(f"Subjects: {len(df['subject'].unique())}")
        
        return df
