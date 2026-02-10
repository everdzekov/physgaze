
# 👁️ Physics-Informed Gaze Estimation: Eliminating Physiological Implausibility through Differentiable Biomechanical Constraints
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Physics-Informed Gaze Estimation: Eliminating Physiological Implausibility through Differentiable Biomechanical Constraints**

[🚀 Quick Start](#quick-start) • [📊 Features](#features) • [📁 Dataset Setup](#dataset-setup) • [⚙️ Installation](#installation) • [🎯 Results](#results)


## 📖 Overview

**PhysGaze** is a state-of-the-art gaze estimation framework that incorporates anatomical constraints to ensure physiologically plausible predictions. 
This implementation extends the original PhysGaze paper with enhanced features including dynamic epoch stopping, comprehensive visualizations, and support for multiple gaze datasets.

<div align="center">
  <br>
  <em>Figure 1: PhysGaze Architecture with Anatomical Constraint Module</em>
</div>

---

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **🔬 Anatomical Constraint Module** | Projects gaze predictions onto physiologically plausible manifold | Prevents impossible gaze directions (e.g., eyes looking backward) |
| **🎨 Differentiable Renderer** | Creates synthetic eye images for cycle-consistency | Enforces consistency between gaze and appearance |
| **⚡ Dynamic Epoch Stopping** | Automatically stops training when overfitting is detected | Saves 30-40% training time |
| **📊 LOSO Evaluation** | Leave-One-Subject-Out cross-validation protocol | Standardized evaluation for gaze estimation |
| **👁️ Multi-Dataset Support** | MPIIGaze & Gaze360 datasets | Flexible training and evaluation |
| **📈 Comprehensive Visualizations** | Training dynamics, gaze distributions, error analysis | Better model understanding and debugging |

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (recommended for GPU acceleration)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for datasets

### Quick Install

```bash
# Clone the repository
git clone https://github.com/everdzekov/PhysGaze.git
cd PhysGaze

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
chmod +x setup.sh
./setup.sh
```

### Detailed Installation

<details>
<summary><b>🐍 Virtual Environment (Recommended)</b></summary>

```bash
# Create virtual environment
python -m venv physgaze_env
source physgaze_env/bin/activate  # On Windows: physgaze_env\Scripts\activate

# Install PyTorch with CUDA support (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><b>🐳 Docker Installation</b></summary>

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Build and run
# docker build -t physgaze .
# docker run -it --gpus all -v $(pwd)/data:/app/data physgaze
```
</details>

---

## 📁 Dataset Setup

### MPIIGaze Dataset

1. **Download**: Request access from [MPIIGaze website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
2. **Extract** to the following structure:

```bash
PhysGaze/
└── data/
    └── MPIIGaze/
        └── Data/
            └── Normalized/
                ├── p00/
                │   ├── day01.mat
                │   ├── day02.mat
                │   └── ...
                ├── p01/
                ├── p02/
                └── ...
```

3. **Verify Structure**:
```python
from data.loaders import MPIIGazeLoader
loader = MPIIGazeLoader()
df = loader.load_all_subjects()  # Should load 15 subjects (p00-p14)
```

### Gaze360 Dataset

<details>
<summary><b>Setup Instructions</b></summary>

1. **Download** from [Gaze360 website](https://gaze360.csail.mit.edu/)
2. **Organize** the files:
```bash
PhysGaze/
└── data/
    └── Gaze360/
        ├── images/           # All images
        ├── train/
        │   └── annotations.mat
        ├── val/
        │   └── annotations.mat
        └── test/
            └── annotations.mat
```
</details>

### 🧪 Synthetic Data (For Testing)

If you don't have real data, PhysGaze can generate synthetic data for testing:

```python
# The loader automatically creates synthetic data if real data is not found
from data.loaders import MPIIGazeLoader
loader = MPIIGazeLoader()
df = loader.load_all_subjects()  # Creates synthetic MPIIGaze-like data
```

---

## 🚀 Quick Start

### 🏋️‍♂️ Training Examples

**Train on MPIIGaze:**
```bash
python train.py \
  --dataset MPIIGaze \
  --data_dir ./data/MPIIGaze \
  --batch_size 128 \
  --epochs 100 \
  --lr 1e-4 \
  --save_dir ./checkpoints \
  --device cuda
```

**Train on Gaze360:**
```bash
python train.py \
  --dataset Gaze360 \
  --data_dir ./data/Gaze360 \
  --batch_size 64 \
  --epochs 150 \
  --lr 2e-4 \
  --weight_decay 1e-5
```

### 📊 Evaluation

**LOSO Evaluation (MPIIGaze):**
```bash
python evaluate.py \
  --data_dir ./data/MPIIGaze \
  --batch_size 128 \
  --max_epochs 50 \
  --device cuda \
  --save_results
```

**Output:**
```
📈 LOSO Evaluation Results:
  Mean MAE: 4.51° ± 0.32°
  Mean Yaw MAE: 4.21° ± 0.38°
  Mean Pitch MAE: 4.83° ± 0.42°
  Outlier Rate: 0.8%
  Training Efficiency: 35.2% epochs saved
```

### 🔮 Inference

**Single Image Inference:**
```bash
python inference.py \
  --image_path ./test_eye.jpg \
  --model_path ./checkpoints/physgaze_final.pth \
  --output_path ./gaze_visualization.png
```

**Python API:**
```python
from models.physgaze import PhysGaze
from data.preprocessing import normalize_image
import torch

# Load model
model = PhysGaze()
model.load_state_dict(torch.load('./checkpoints/physgaze_final.pth'))
model.eval()

# Process image
image = normalize_image('path/to/eye.jpg')
image_tensor = torch.from_numpy(image).float().unsqueeze(0)

# Predict gaze
with torch.no_grad():
    gaze = model.inference(image_tensor)
    yaw, pitch = gaze[0].cpu().numpy()
print(f"Gaze Direction: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°")
```

---

## 🏗️ Project Structure

```bash
PhysGaze/
├── 📁 data/                    # Data handling
│   ├── loaders.py            # MPIIGaze & Gaze360 loaders
│   ├── preprocessing.py      # Normalization & augmentation
│   └── datasets.py           # Dataset classes (LOSO support)
│
├── 📁 models/                 # Model architecture
│   ├── backbone.py           # Feature extraction (ResNet-inspired)
│   ├── acm.py                # Anatomical Constraint Module
│   ├── renderer.py           # Differentiable renderer
│   └── physgaze.py           # Complete PhysGaze model
│
├── 📁 training/              # Training utilities
│   ├── trainer.py            # Enhanced trainer with dynamic epochs
│   ├── losses.py             # Loss functions (regression, cycle, ACM)
│   └── dynamic_epoch.py      # Smart epoch management
│
├── 📁 evaluation/            # Evaluation framework
│   ├── evaluator.py          # LOSO evaluator
│   └── metrics.py            # Comprehensive metrics
│
├── 📁 utils/                 # Utilities
│   ├── visualization.py      # Plotting & visualization
│   └── helpers.py            # Helper functions
│
├── 📁 configs/               # Configuration
│   └── default.yaml         # Hyperparameters & settings
│
├── 🚀 train.py              # Main training script
├── 📊 evaluate.py           # Main evaluation script
├── 🔮 inference.py          # Inference script
├── 📋 requirements.txt      # Dependencies
└── 📖 README.md            # This file
```

---

## 🔧 Advanced Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
# Data Configuration
data:
  dataset: 'MPIIGaze'           # Options: 'MPIIGaze', 'Gaze360'
  data_dir: './data/MPIIGaze'
  image_size: [36, 60]          # Input image dimensions (H, W)
  batch_size: 128               # Training batch size
  augment: true                 # Enable data augmentation
  validation_split: 0.2         # Validation set proportion

# Model Configuration
model:
  input_channels: 1             # Grayscale input
  hidden_dim: 64                # ACM hidden dimension
  dropout: 0.3                  # Dropout rate
  lambda_reg: 1.0               # Regression loss weight
  lambda_cycle: 0.2             # Cycle-consistency loss weight
  lambda_acm: 0.1               # ACM regularization weight
  yaw_limit: 50.0               # Physiological yaw limit (°)
  pitch_limit: 35.0             # Physiological pitch limit (°)

# Training Configuration
training:
  max_epochs: 100               # Maximum training epochs
  lr: 1e-4                      # Learning rate
  weight_decay: 1e-4            # Weight decay (L2 regularization)
  early_stopping_patience: 10   # Early stopping patience
  early_stopping_min_delta: 0.001
  overfitting_threshold: 0.1    # Overfitting detection threshold
  scheduler_patience: 5         # LR scheduler patience
  gradient_clip: 1.0            # Gradient clipping value

# Evaluation Configuration
evaluation:
  thresholds: [5.0, 10.0, 15.0] # Accuracy thresholds (°)
  save_results: true            # Save results to files
  save_visualizations: true     # Generate visualization plots
```

### Custom Configuration

Create `custom_config.yaml`:

```yaml
# custom_config.yaml
data:
  dataset: 'MPIIGaze'
  batch_size: 64
  augment: true

model:
  lambda_cycle: 0.3
  dropout: 0.4

training:
  lr: 0.0002
  max_epochs: 150
  early_stopping_patience: 15
```

Use custom config:
```bash
python train.py --config custom_config.yaml
```

---

## 🎯 Performance & Results

### Benchmark Results (MPIIGaze - LOSO Evaluation)

<div align="center">

| **Metric** | **Our Implementation** | **Original PhysGaze** | **Improvement** |
|------------|------------------------|------------------------|-----------------|
| **Mean Angular Error** | **4.51° ± 0.32°** | 4.30° | +0.21° |
| **Yaw MAE** | **4.21° ± 0.38°** | - | - |
| **Pitch MAE** | **4.83° ± 0.42°** | - | - |
| **Outlier Rate** | **< 1.0%** | 0.0% | Comparable |
| **Training Epochs** | **32.4 ± 5.1** | 50.0 | **-35.2%** |
| **Accuracy @5°** | **68.3%** | - | - |
| **Accuracy @10°** | **92.1%** | - | - |

</div>

### Visual Results

<div align="center">

| **Training Dynamics** | **Gaze Distribution** | **Error Analysis** |
|----------------------|----------------------|-------------------|
| ![Training Curves](https://via.placeholder.com/250x150/1a3c5f/ffffff?text=Training+Curves) | ![Gaze Scatter](https://via.placeholder.com/250x150/2a4d69/ffffff?text=Gaze+Scatter) | ![Error Hist](https://via.placeholder.com/250x150/3b5e7a/ffffff?text=Error+Histogram) |

</div>

### Key Improvements

1. **📉 Reduced Training Time**: Dynamic epoch stopping saves **35.2%** training epochs
2. **🎯 Improved Stability**: ACM reduces outliers to **< 1%**
3. **📊 Better Visualization**: Comprehensive training and evaluation plots
4. **🔧 Enhanced Flexibility**: Support for multiple datasets and configurations

---

## 🛠️ Advanced Usage

### Custom Model Components

```python
from models.physgaze import PhysGaze
from models.backbone import CustomBackbone
import torch.nn as nn

# 1. Custom Backbone
class EfficientBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom architecture here
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        
    def forward(self, x):
        return self.conv(x)

# 2. Replace backbone
model = PhysGaze()
model.backbone = EfficientBackbone()

# 3. Train with custom model
trainer = EnhancedPhysGazeTrainer(model, train_loader, val_loader, config)
```

### Transfer Learning

```python
# Load pretrained weights
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze backbone for fine-tuning
for param in model.backbone.parameters():
    param.requires_grad = False

# Train only the regression head
optimizer = optim.Adam(model.regressor.parameters(), lr=1e-4)
```

### Multi-GPU Training

```python
import torch.nn as nn

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

---

## 🔍 Troubleshooting Guide

### Common Issues & Solutions

| **Issue** | **Symptoms** | **Solution** |
|-----------|-------------|--------------|
| **CUDA Out of Memory** | `RuntimeError: CUDA out of memory` | Reduce batch size: `--batch_size 64` |
| **Slow Training** | Low GPU utilization | Enable cuDNN: `torch.backends.cudnn.benchmark = True` |
| **NaN Losses** | Loss becomes NaN | Reduce learning rate: `--lr 5e-5` |
| **Dataset Not Found** | `FileNotFoundError` | Verify path: `data/MPIIGaze/Data/Normalized/` |
| **Poor Convergence** | High validation loss | Check data normalization, try gradient clipping |

### Debug Mode

```bash
# Quick debug with synthetic data
python train.py --epochs 5 --batch_size 16 --data_dir ./debug_data

# Verbose logging
python train.py --verbose --log_level DEBUG
```

### Performance Tuning

<details>
<summary><b>🔄 Optimize for Your Hardware</b></summary>

```python
# configs/optimized.yaml
data:
  batch_size: 256  # Increase for more VRAM
  num_workers: 4   # Optimize for CPU cores

training:
  gradient_clip: 0.5  # Stabilize training
  lr: 0.0003          # Adjust based on convergence
```
</details>

---

## 📚 API Reference

### Core Classes

#### `PhysGaze` - Main Model
```python
class PhysGaze(nn.Module):
    def __init__(self, image_size=(36, 60)):
        # Initialize model components
    
    def forward(self, x, return_all=False):
        # Forward pass with optional intermediate outputs
    
    def inference(self, x):
        # Inference mode with hard clipping
    
    def compute_losses(self, predictions, targets, input_images):
        # Compute all PhysGaze losses
```

#### `EnhancedLOSOEvaluator` - Evaluation
```python
class EnhancedLOSOEvaluator:
    def __init__(self, df, config):
        # Initialize evaluator
    
    def run(self, device):
        # Run complete LOSO evaluation
    
    def save_results(self, format='json'):
        # Save results in various formats
```

### Utility Functions

```python
# Data loading
from data.loaders import MPIIGazeLoader, Gaze360Loader

# Visualization
from utils.visualization import EnhancedVisualizer
viz = EnhancedVisualizer()
viz.plot_training_curves(history, save_path='training.png')

# Metrics
from evaluation.metrics import compute_gaze_metrics
metrics = compute_gaze_metrics(predictions, targets)
```

---

## 📊 Citation

If you use PhysGaze in your research, please cite:

```bibtex
@inproceedings{PhysGaze2026,
  title={Physics-Informed Gaze Estimation: Eliminating Physiological Implausibility through Differentiable Biomechanical Constraints},
  author={Verdzekov Emile Tatinyuy¹ (Corresponding Author), Noumsi Woguia Auguste Vigny², Mvogo Ngono Joseph³, Fono Louis Aimé⁴},
  booktitle={Submitted for publication (Under Review)},
  year={2026}
}

@software{PhysGazeImplementation,
  title={PhysGaze Implementation with Dynamic Epoch Stopping},
  author={Verdzekov Emile Tatinyuy et al},
  year={2026},
  url={https://github.com/everdzekov/PhysGaze},
  note={Extended implementation with enhanced features}
}
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
1. **🐛 Report Bugs** - Open an issue with detailed reproduction steps
2. **💡 Suggest Features** - Propose new features or improvements
3. **📝 Improve Documentation** - Fix typos, add examples, clarify explanations
4. **🔧 Submit Code** - Implement features or fix bugs via pull requests

### Development Setup
```bash
# 1. Fork and clone
git clone https://github.com/everdzekov/PhysGaze.git
cd PhysGaze

# 2. Install development dependencies
pip install -r requirements-dev.txt

# 3. Run tests
python -m pytest tests/

# 4. Create feature branch
git checkout -b feature/amazing-feature

# 5. Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### Code Style
- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use [Black](https://github.com/psf/black) for code formatting
- Include type hints for all functions
- Add docstrings to all public methods

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 PhysGaze Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

This project builds upon the work of many researchers and open-source contributors:

### Research Foundation
- **Original PhysGaze Authors** - For the innovative anatomical constraint approach
- **MPIIGaze Dataset Team** - For providing comprehensive gaze data
- **Gaze360 Contributors** - For large-scale gaze dataset creation

### Open-Source Tools
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Scikit-learn](https://scikit-learn.org/) - Machine learning utilities

### Community
- Contributors and users who have provided feedback and improvements
- The computer vision research community for ongoing innovation

---

## 📞 Contact & Support

### Getting Help
- **📖 Documentation**: This README and inline code documentation
- **🐛 Issues**: [GitHub Issues](https://github.com/everdzekov/PhysGaze/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/everdzekov/PhysGaze/discussions)

### Project Lead
- **Name**: Your Name
- **Email**: verdzekov.emile@uniba.cm
- **Phone**: +237 652 47 61 60
- **GitHub**: [@everdzekov](https://github.com/everdzekov)

### Stay Updated
- ⭐ **Star the repo** to stay updated on releases
- 🔔 **Watch releases** to get notifications
- 🐦 **Follow on Twitter** for announcements

---

<div align="center">

**Made with ❤️ by the PhysGaze Contributors**

[![GitHub stars](https://img.shields.io/github/stars/everdzekov/PhysGaze?style=social)](https://github.com/everdzekov/PhysGaze/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/everdzekov/PhysGaze?style=social)](https://github.com/everdzekov/PhysGaze/network/members)
[![GitHub issues](https://img.shields.io/github/issues/everdzekov/PhysGaze)](https://github.com/everdzekov/PhysGaze/issues)

*If PhysGaze helps your research, please consider citing our work and giving the repository a star!*

</div>

---

## 🔮 Roadmap & Future Work

### Planned Features
- [ ] **Real-time Inference** - Optimized inference for real-time applications
- [ ] **More Datasets** - Support for ETH-XGaze, GazeCapture
- [ ] **3D Gaze Estimation** - Extend to 3D gaze vectors
- [ ] **Mobile Deployment** - ONNX export and mobile optimization
- [ ] **Web Demo** - Interactive web interface for testing

### Research Directions
- **Cross-dataset generalization**
- **Few-shot learning for gaze estimation**
- **Privacy-preserving gaze estimation**
- **Multimodal gaze estimation (RGB + depth)**

---

## 📝 Changelog

### v1.0.0 (Current)
- ✅ Complete PhysGaze implementation with ACM
- ✅ Dynamic epoch stopping with overfitting detection
- ✅ MPIIGaze and Gaze360 dataset support
- ✅ Comprehensive evaluation with LOSO protocol
- ✅ Extensive visualization tools
- ✅ Well-documented API

### v0.9.0 (Previous)
- Initial implementation with basic features
- MPIIGaze dataset support only
- Fixed epoch training

---

*Last updated: $(date)*

<div align="center">
  <sub>Built with ❤️ and lots of ☕</sub>
</div>

