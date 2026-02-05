import os
import numpy as np
from pathlib import Path
import scipy.io as sio
import urllib.request
import tarfile
from typing import Tuple, Dict, List, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
        debug: bool = False
    ):
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
        self._load_dataset()

    def _load_dataset(self):
        """Load the dataset from processed files or raw data."""
        processed_file = self.root_dir / f'processed_{self.split}.pt'

        if processed_file.exists():
            try:
                data = torch.load(processed_file)
                self.samples = data['samples']
            except:
                self._process_raw_data()
        else:
            self._process_raw_data()

    def _process_raw_data(self):
        """Process raw MPIIGaze data."""
        for subject_id in self.subject_ids:
            subject_dir = self.root_dir / 'Data' / 'Normalized' / f'p{subject_id:02d}'
            
            if not subject_dir.exists():
                continue

            mat_files = list(subject_dir.glob('*.mat'))
            
            for mat_file in mat_files:
                try:
                    mat_data = sio.loadmat(str(mat_file))
                    
                    if 'data' in mat_data:
                        data = mat_data['data']
                        
                        for i in range(data.shape[1]):
                            sample = data[0, i]
                            
                            if 'left' not in sample.dtype.names or 'right' not in sample.dtype.names:
                                continue

                            # Left eye
                            left_eye = sample['left'][0, 0]['image'][0, 0]
                            left_gaze = sample['left'][0, 0]['gaze'][0, 0].flatten()
                            
                            # Right eye
                            right_eye = sample['right'][0, 0]['image'][0, 0]
                            right_gaze = sample['right'][0, 0]['gaze'][0, 0].flatten()

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
                            
                except Exception as e:
                    continue

        # Save processed data
        if len(self.samples) > 0:
            processed_file = self.root_dir / f'processed_{self.split}.pt'
            torch.save({'samples': self.samples}, processed_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = sample['image']
        gaze = sample['gaze']

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        elif not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Ensure image is [C, H, W] format
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        elif len(image.shape) == 3 and image.shape[-1] in [1, 3]:
            image = image.permute(2, 0, 1)

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
            yaw = torch.atan2(gaze_tensor[0], -gaze_tensor[2]) * 180 / np.pi
            pitch = torch.asin(gaze_tensor[1]) * 180 / np.pi
            gaze_tensor = torch.stack([yaw, pitch])

        # Normalize gaze angles to [-1, 1]
        if self.normalize_gaze:
            gaze_tensor[0] = gaze_tensor[0] / self.yaw_max
            gaze_tensor[1] = gaze_tensor[1] / self.pitch_max
            gaze_tensor = torch.clamp(gaze_tensor, -1, 1)

        if self.transform:
            image = self.transform(image)

        return image, gaze_tensor


def download_mpiigaze(root_dir: str = './data/MPIIGaze') -> bool:
    """
    Download and extract the MPIIGaze dataset.
    """
    url = "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz"
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    tar_file = root_path / "MPIIGaze.tar.gz"

    if (root_path / 'Data').exists():
        print("MPIIGaze dataset already exists.")
        return True

    print(f"Downloading MPIIGaze dataset from {url}...")
    
    try:
        urllib.request.urlretrieve(url, tar_file)
        print("Download complete!")
        
        print("Extracting dataset...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(root_path)
        
        os.remove(tar_file)
        print("Extraction complete!")
        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False