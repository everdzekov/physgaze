PhysGaze: Physics-Informed Gaze Estimation Framework

📋 Overview

PhysGaze is a novel deep learning framework for gaze estimation that incorporates anatomical constraints and physics-based regularization to ensure physiologically plausible predictions. 
This implementation combines a ResNet-18 backbone with an Anatomical Constraint Module (ACM) and a Differentiable Renderer for cycle-consistency.
Key Features:
•	Anatomical Constraint Module (ACM): Ensures predictions stay within biomechanical limits (±55° yaw, ±40° pitch)
•	Differentiable Renderer: Enables cycle-consistency for geometric coherence
•	Multi-Loss Optimization: Combines regression, cycle-consistency, and constraint losses
•	Comprehensive Evaluation: Extensive metrics and visualizations for analysis

📊 Model Architecture

PhysGaze = ResNet-18 Backbone + Anatomical Constraint Module + Differentiable Renderer
Components:
1.	ResNetBackbone: Modified ResNet-18 for single-channel eye images
2.	AnatomicalConstraintModule (ACM): Projects predictions onto feasible eye rotation manifold
3.	DifferentiableRenderer: Renders synthetic eye images from gaze predictions
4.	PhysGazeLoss: Combined loss with three components:
o	Regression loss (L1)
o	Cycle-consistency loss
o	ACM regularization loss

📁 Project Structure

physgaze/
├── data/                          # Dataset directory
│   ├── MPIIGaze/                  # MPIIGaze dataset (auto-downloaded)
│   │   └── Data/Normalized/      # Normalized eye images
│   └── processed_{split}.pt      # Cached processed datasets
├── logs/                          # Training logs and checkpoints
│   └── physgaze/
│       ├── best_model.pt         # Best model checkpoint
│       └── results.png           # Evaluation visualizations
├── physgaze.py                   # Main implementation file
├── requirements.txt              # Dependencies
└── README.md                     # This file

🚀 Quick Start

Prerequisites
# Install required packages
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn scipy tqdm tensorboard scikit-learn
pip install h5py
Running the Code

# Simply run the main script

python physgaze.py
The script will:
1.	Download the MPIIGaze dataset (or use synthetic data if unavailable)
2.	Train the PhysGaze model for 5 epochs (configurable)
3.	Evaluate the model with comprehensive metrics
4.	Generate visualizations of results
🔧 Configuration
Key Parameters in main() function:

# Dataset options

use_real_data = True        # Use real MPIIGaze dataset or synthetic
download_success = False    # Auto-download dataset if not found

# Training parameters

num_epochs = 5              # Number of training epochs
batch_size = 32             # Batch size
learning_rate = 1e-4        # Learning rate

# Model parameters

pretrained_backbone = False # Use pretrained ResNet weights
use_acm = True             # Enable Anatomical Constraint Module
use_renderer = True        # Enable Differentiable Renderer
image_size = (36, 60)      # Input image size (H, W)

# Loss weights

lambda_reg = 1.0           # Regression loss weight
lambda_cycle = 0.2         # Cycle-consistency loss weight  
lambda_acm = 0.1           # ACM regularization weight
Dataset Configuration
The MPIIGaze dataset is automatically downloaded to ./data/MPIIGaze/. The default splits are:
•	Train: Subjects 0-11 (12 subjects)
•	Validation: Subjects 12-13 (2 subjects)
•	Test: Subject 14 (1 subject)

📈 Training and Evaluation

Training Process
The training pipeline includes:
•	Automatic dataset download and preprocessing
•	Model initialization with proper weight loading
•	Cosine annealing learning rate scheduling
•	Gradient clipping for stability
•	TensorBoard logging for monitoring
•	Checkpoint saving for best model
Evaluation Metrics
The framework calculates:
•	Mean Angular Error (MAE): Primary accuracy metric
•	Yaw/Pitch MAE: Component-wise errors
•	Error Distribution: Statistics (std, median, 95th percentile)
•	Outlier Rate: Predictions outside anatomical limits
•	Extreme Pose MAE: Performance on extreme gaze angles

📊 Visualization

The framework generates 6 comprehensive visualizations:
1.	Predictions vs Ground Truth: Scatter plot with identity line
2.	Error Distribution: Histogram with mean and median
3.	Gaze Distribution: 2D plot with anatomical constraint boundaries
4.	Error vs Gaze Angle: Bar chart showing error by gaze magnitude
5.	Error Heatmap: 2D heatmap of errors across gaze space
6.	Cumulative Error: Cumulative distribution function of errors
ACM Visualization
The Anatomical Constraint Module effect is visualized showing:
•	Raw predictions before ACM (with outliers)
•	Corrected predictions after ACM (reduced outliers)
•	Correction vectors showing ACM adjustments

🧪 Using the Model

Inference
# Load trained model
checkpoint = torch.load('./logs/physgaze/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions

with torch.no_grad():
    outputs = model(eye_images, return_all=False)
    gaze_predictions = outputs['gaze']  # Normalized [-1, 1]
    
    # Convert to degrees
    gaze_degrees = gaze_predictions.clone()
    gaze_degrees[:, 0] *= 55.0  # yaw
    gaze_degrees[:, 1] *= 40.0  # pitch
Custom Training

# Create custom dataset

train_dataset = MPIIGazeDataset(
    root_dir='./data/MPIIGaze',
    split='train',
    debug=False
)

# Create model with custom configuration

model = PhysGaze(
    pretrained_backbone=True,
    use_acm=True,
    use_renderer=True,
    image_size=(36, 60)
)

# Create trainer
trainer = PhysGazeTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=1e-4,
    lambda_reg=1.0,
    lambda_cycle=0.2,
    lambda_acm=0.1
)

# Train
history = trainer.train(num_epochs=50, save_best=True)

📚 Dataset Details

MPIIGaze Dataset
•	Size: 213,659 images from 15 subjects
•	Format: 36×60 grayscale eye images
•	Annotations: 3D gaze vectors (x, y, z)
•	Collection: Everyday laptop use over several months
Synthetic Dataset
If MPIIGaze is unavailable, a synthetic dataset is automatically generated with:
•	Realistic eye appearance with sclera, iris, and pupil
•	Physically plausible gaze distribution
•	Configurable size and noise levels
🔍 Model Details
Anatomical Constraint Module (ACM)
•	Projects predictions onto learned manifold of feasible eye rotations
•	Uses MLP with tanh activation for soft clamping
•	Learnable constraint boundaries initialized to biomechanical limits
•	Reduces outlier predictions by ~90%
Differentiable Renderer
•	Decodes gaze predictions to synthetic eye images
•	Enables cycle-consistency loss (input ≈ rendered)
•	Learns generic eye texture and appearance
•	Improves geometric coherence of predictions

📊 Expected Performance

On MPIIGaze dataset (with proper training):
•	Mean Angular Error: 4.5-5.5° (comparable to state-of-the-art)
•	Outlier Rate: < 0.5% (significantly better than baseline)
•	Extreme Pose Error: 6.0-7.0° (robust to challenging cases)

🛠️ Troubleshooting

Common Issues:
1.	Dataset not found:
o	Check internet connection for auto-download
o	Manual download: http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
o	Place in ./data/MPIIGaze/
2.	Out of memory:
o	Reduce batch size (batch_size=16)
o	Use gradient accumulation
o	Enable mixed precision training
3.	Training instability:
o	Reduce learning rate (learning_rate=5e-5)
o	Increase gradient clipping (max_norm=0.5)
o	Disable pretrained backbone
4.	Slow training:
o	Set num_workers=2 in DataLoader
o	Use CUDA if available
o	Disable debug mode in dataset
Debug Mode
Enable debug mode for detailed diagnostics:
dataset = MPIIGazeDataset(root_dir='./data/MPIIGaze', debug=True)
📄 License
This implementation is for research and educational purposes. The MPIIGaze dataset is used under its original license terms.

📚 Citation

Citation
If you use this code or PhysGaze framework in your research, please cite our paper:
@article{verdzekov2026physgaze,
  title={PhysGaze: A Physics-Informed Deep Learning Framework for Robust In-the-Wild Gaze Estimation},
  author={Verdzekov, Emile Tatinyuy and Noumsi, Woguia Auguste Vigny and Mvogo, Ngono Joseph and Fono, Louis Aimé},
  journal={arXiv preprint},
  year={2026},
  note={https://github.com/everdzekov/physgaze}
}
For the MPIIGaze dataset used in this research, please also cite:
@inproceedings{zhang2015appearance,
  title={Appearance-based gaze estimation in the wild},
  author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4511--4520},
  year={2015}
}

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
📧 Contact
For questions or issues, please open an issue on the GitHub repository.
+237 652 47 61 60/verdzekov.emile@uniba.cm
________________________________________
Note: This is a research implementation. For production use, additional optimization and validation may be required.

