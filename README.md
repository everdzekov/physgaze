# PhysGaze: Physiologically Constrained Gaze Estimation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PhysGaze is a deep learning framework for gaze estimation that incorporates anatomical constraints to ensure physiologically plausible predictions. 
This implementation follows the PhysGaze paper with enhanced features like dynamic epoch stopping and comprehensive visualizations.

## Features

- **Anatomical Constraint Module (ACM)**: Projects gaze predictions onto physiologically plausible manifold
- **Differentiable Renderer**: Enforces cycle consistency between gaze and eye appearance
- **Dynamic Epoch Stopping**: Automatically stops training when overfitting is detected
- **Leave-One-Subject-Out (LOSO) Evaluation**: Standard evaluation protocol for gaze estimation
- **Comprehensive Visualizations**: Training dynamics, gaze distributions, and error analysis
- **Multi-Dataset Support**: MPIIGaze and Gaze360 datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/everdzekov/PhysGaze.git
cd PhysGaze
