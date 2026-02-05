import numpy as np
from typing import Tuple, List, Dict
import torch


class SyntheticMPIIGaze:
    """
    Synthetic MPIIGaze-like dataset for testing without downloading the real data.
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
        
        self.yaw_max = 55.0
        self.pitch_max = 40.0
        
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict]:
        samples = []
        
        for _ in range(self.num_samples):
            yaw = np.random.uniform(-50, 50)
            pitch = np.random.uniform(-35, 25)
            image = self._render_eye(yaw, pitch)
            
            samples.append({
                'image': image,
                'gaze': np.array([yaw, pitch])
            })
        
        return samples
    
    def _render_eye(self, yaw: float, pitch: float) -> np.ndarray:
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.float32)
        
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        
        # Eye white
        sclera_a, sclera_b = w // 3, h // 3
        sclera_mask = ((x - cx) / sclera_a) ** 2 + ((y - cy) / sclera_b) ** 2 <= 1
        image[sclera_mask] = 0.9 + np.random.uniform(-0.05, 0.05)
        
        # Iris position
        iris_offset_x = (yaw / self.yaw_max) * (w // 6)
        iris_offset_y = (pitch / self.pitch_max) * (h // 6)
        iris_cx = cx + iris_offset_x
        iris_cy = cy + iris_offset_y
        
        # Iris
        iris_r = h // 4
        iris_mask = (x - iris_cx) ** 2 + (y - iris_cy) ** 2 <= iris_r ** 2
        iris_mask = iris_mask & sclera_mask
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
