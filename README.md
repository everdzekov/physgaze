PhysGaze: A Physics-Informed Deep Learning Framework for Robust In-the-Wild Gaze Estimation
📖 Overview
PhysGaze is a novel deep learning framework that pioneers a paradigm shift in gaze estimation by explicitly integrating biomechanical knowledge through differentiable physics models. Unlike traditional appearance-based methods that frequently generate physiologically impossible predictions, PhysGaze ensures both accuracy and physical plausibility through two key innovations:
1.	Differentiable Anatomical Constraint Module (ACM): Actively corrects implausible gaze predictions onto a learned manifold of physiologically feasible eye rotations
2.	Differentiable Renderer (DR): Enforces geometric consistency between predicted gaze directions and input images through a cycle-consistency loss
This synergistic integration achieves state-of-the-art performance on standard benchmarks while completely eliminating anatomical outliers—a critical advancement for real-world deployment.
🏆 Key Achievements
•	Mean Angular Error: 4.3° on MPIIGaze (19.6% improvement over previous methods)
•	Outlier Reduction: 0.0% physiological outliers (vs. 12.4% in baseline models)
•	Extreme Pose Robustness: 42.1% error reduction on head poses > 40°
•	Real-time Inference: ~8 ms per frame on standard hardware
🏗️ Framework Architecture
Core Components
PhysGaze Pipeline:
Input Image → Backbone Network → ACM → DR → Final Prediction
                      		 ↓                              ↓                         ↓
                                 Initial Gaze → Constrained → Rendered
                                   Prediction       Prediction         Image
                                       ↓                                                ↑
                                       L_reg ← Cycle → L_cycle
                                        Loss    Consistency  Loss
Feature Extraction Backbone: Lightweight ResNet-18 adapted for single-channel eye images (224×224)
Anatomical Constraint Module (ACM):
•	Differentiable MLP that learns smooth projection onto physiological manifold
•	Respects coupled yaw-pitch constraints (±55° yaw, ±40° pitch limits)
•	Eliminates implausible predictions while preserving valid ones
Differentiable Renderer (DR):
•	Spherical eye model (12mm radius) with learnable UV texture
•	Phong shading with camera-aligned light source
•	Enables self-supervised cycle-consistency training
Loss Functions
The model is trained end-to-end with a multi-objective loss:
L_total = λ_reg·L_reg + λ_cycle·L_cycle + λ_acm·L_reg_acm
•	L_reg: Mean Absolute Error between predicted and ground truth gaze angles
•	L_cycle: L1 distance between input and rendered images (cycle-consistency)
•	L_reg_acm: L2 regularization on ACM corrections to prevent over-correction
📊 Performance Comparison
MPIIGaze Dataset
Method	Mean Angular Error (°)	Outlier Rate (%)
Baseline (ResNet-18)	5.5	12.4
RT-GENE	5.1	8.2
ETH-XGaze	5.3	6.8
PhysGaze (Ours)	4.2	0.0




Gaze360 Dataset (Full-Sphere Gaze)
Method	Mean Angular Error (°)
Gaze360	8.5
ETH-XGaze	8.3
PhysGaze (Ours)	7.6
🚀 Quick Start
Installation
# Clone the repository
git clone https://github.com/everdzekov/physgaze.git
cd physgaze

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch3D (requires separate installation)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
Dataset Setup
1.	MPIIGaze:
# Download and extract dataset
mkdir -p data/MPIIGaze
cd data/MPIIGaze
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
tar -xzf MPIIGaze.tar.gz
2.	Gaze360:
# Clone and setup Gaze360
git clone https://github.com/erkil1452/gaze360.git data/Gaze360
cd data/Gaze360
python download_dataset.py  # Follow instructions in repository
Training
# Train on MPIIGaze with default configuration
python train_mpiigaze.py

# Train on Gaze360
python train_gaze360.py

# With custom configuration
python train.py --config configs/mpiigaze_config.yaml \
                --batch_size 128 \
                --epochs 50 \
                --lr 1e-4
Evaluation
# Evaluate on test set
python evaluate.py --model checkpoints/best_model.pth \
                   --dataset mpiigaze \
                   --split test

# Generate performance report
python evaluate.py --model checkpoints/best_model.pth \
                   --output results/evaluation_report.json

Inference
# Run inference on single image
python inference.py --image path/to/eye_image.jpg \
                    --model checkpoints/best_model.pth \
                    --visualize

# Real-time webcam gaze estimation
python inference.py --webcam \
                    --model checkpoints/best_model.pth \
                    --fps 30
📁 Project Structure
PhysGaze/
├── configs/                    # Configuration files
│   ├── mpiigaze_config.yaml   # MPIIGaze training config
│   ├── gaze360_config.yaml    # Gaze360 training config
│   └── inference_config.yaml  # Inference settings
├── data/                      # Data loading modules
│   ├── mpiigaze_dataset.py    # MPIIGaze Dataset class
│   ├── gaze360_dataset.py     # Gaze360 Dataset class
│   ├── preprocessing.py       # Image preprocessing
│   └── augmentations.py       # Data augmentations
├── models/                    # Model definitions
│   ├── backbone.py           # Feature extraction networks
│   ├── acm.py               # Anatomical Constraint Module
│   ├── renderer.py          # Differentiable Renderer
│   └── physgaze.py          # Complete PhysGaze model
├── training/                 # Training utilities
│   ├── trainer.py           # Main training loop
│   ├── losses.py            # Loss functions
│   └── optimizers.py        # Optimizer configurations
├── evaluation/              # Evaluation scripts
│   ├── metrics.py           # Evaluation metrics
│   ├── visualize.py         # Result visualization
│   └── analyze.py           # Statistical analysis
├── utils/                   # Utility functions
│   ├── gaze_utils.py        # Gaze conversion utilities
│   ├── geometry.py          # Geometric transformations
│   └── io_utils.py          # File I/O helpers
├── checkpoints/             # Model checkpoints
├── outputs/                 # Training outputs
│   ├── logs/               # Training logs
│   ├── figures/            # Generated figures
│   └── results/            # Evaluation results
├── experiments/            # Experiment configurations
├── docs/                  # Documentation
├── tests/                 # Unit tests
├── train.py               # Main training script
├── evaluate.py            # Main evaluation script
├── inference.py           # Inference script
├── requirements.txt       # Python dependencies
└── setup.py              # Package setup



⚙️ Configuration
Main Configuration Parameters
# configs/mpiigaze_config.yaml
model:
  backbone: "resnet18"        # Feature extractor
  input_channels: 1           # Grayscale input
  use_acm: true              # Enable Anatomical Constraint Module
  use_renderer: true         # Enable Differentiable Renderer

training:
  batch_size: 128
  epochs: 50
  learning_rate: 1e-4
  optimizer: "AdamW"
  weight_decay: 1e-4

loss:
  lambda_reg: 1.0            # Regression loss weight
  lambda_cycle: 0.2          # Cycle-consistency loss weight
  lambda_acm: 0.1            # ACM regularization weight

renderer:
  eye_radius: 12.0           # Eyeball radius in mm
  texture_size: 512          # UV texture resolution
  shading: "phong"           # Shading model


📈 Results Visualization
Training Progress
# Monitor training with TensorBoard
tensorboard --logdir outputs/logs/

# Generate training plots
python utils/visualize_training.py --log_dir outputs/logs/ --output_dir outputs/figures/
Qualitative Results
# Generate qualitative comparison
python evaluation/visualize.py --model checkpoints/best_model.pth \
                               --dataset mpiigaze \
                               --num_samples 10 \
                               --output_dir outputs/qualitative/
🧪 Ablation Study Results
Model Variant	ACM	DR	MAE (°)	Outliers (%)
Baseline	✗	✗	5.5	12.4
+ ACM only	✓	✗	5.0	0.5
+ DR only	✗	✓	4.9	10.1
PhysGaze	✓	✓	4.3	0.0
Note: ACM = Anatomical Constraint Module, DR = Differentiable Renderer
🎯 Key Features
1. Physiological Plausibility Guarantee
•	Hard elimination of anatomically impossible predictions
•	Learned manifold respects coupled yaw-pitch constraints
•	Smooth, differentiable correction preserving gradient flow
2. Geometric Consistency
•	Self-supervised cycle-consistency learning
•	Differentiable rendering for end-to-end training
•	No additional labeled data required
3. Extreme Condition Robustness
•	42.1% error reduction on extreme head poses
•	Stable performance across lighting variations
•	Generalizes well to unseen subjects
4. Real-time Capable
•	< 0.3 ms inference overhead on GPU
•	~8 ms per frame on CPU
•	Suitable for mobile deployment
🔬 Advanced Usage
Custom Training
from models.physgaze import PhysGaze
from training.trainer import PhysGazeTrainer

# Initialize model
model = PhysGaze(
    backbone_type='resnet18',
    use_acm=True,
    use_renderer=True,
    texture_size=512
)

# Custom training configuration
trainer = PhysGazeTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lambda_reg=1.0,
    lambda_cycle=0.2,
    lambda_acm=0.1,
    device='cuda'
)

# Train with custom callbacks
trainer.train(
    epochs=100,
    save_dir='checkpoints/custom',
    log_dir='logs/custom',
    early_stopping_patience=20
)
Extending the Framework
# Adding new backbone
from models.backbone import CustomBackbone

class CustomPhysGaze(PhysGaze):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = CustomBackbone()  # Replace with custom architecture
        
# Adding new loss functions
from training.losses import CustomLoss

trainer.loss_functions['custom'] = CustomLoss(weight=0.5)
📊 Citation
If you use PhysGaze in your research, please cite:
@article{verdzekov2026physgaze,
  title={PhysGaze: A Physics-Informed Deep Learning Framework for Robust In-the-Wild Gaze Estimation},
  author={Verdzekov, Emile Tatinyuy and Noumsi, Woguia Auguste Vigny and Mvogo, Ngono Joseph and Fono, Louis Aim{\'e}},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/everdzekov/physgaze}
}
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
1.	Fork the repository
2.	Create a feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request
Development Setup
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .
flake8 .
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📞 Contact
•	Corresponding Author: Verdzekov Emile Tatinyuy (apolokange@yahoo.com)
•	GitHub Issues: https://github.com/everdzekov/physgaze/issues
•	Repository: https://github.com/everdzekov/physgaze
🙏 Acknowledgments
•	University of Douala Department of Applied Computer Science
•	Open-source libraries: PyTorch, PyTorch3D, OpenCV
•	Dataset providers: MPIIGaze and Gaze360 teams
•	The open-source research community
________________________________________
This README corresponds to the PhysGaze framework described in the paper "PhysGaze: A Physics-Informed Deep Learning Framework for Robust In-the-Wild Gaze Estimation" by Verdzekov et al. (2026).

