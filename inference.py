#!/usr/bin/env python3
"""
PhysGaze Inference Script
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.physgaze import PhysGaze
from data.preprocessing import normalize_image
from utils.helpers import get_device

def parse_args():
    parser = argparse.ArgumentParser(description='PhysGaze inference on single image')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input eye image')
    parser.add_argument('--model_path', type=str, default='./checkpoints/physgaze_final.pth',
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
    return parser.parse_args()

def load_image(image_path):
    """Load and preprocess image"""
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Try different loading methods
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(img)
    except:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Normalize
    img = normalize_image(img, target_size=(36, 60))
    
    return img

def visualize_gaze(image, yaw, pitch, save_path=None):
    """Visualize gaze direction on image"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Input Eye Image')
    axes[0].axis('off')
    
    # Gaze visualization
    axes[1].imshow(np.ones((200, 200, 3)) * 0.9)
    
    # Draw coordinate system
    center = (100, 100)
    axes[1].arrow(center[0], center[1], 80, 0, head_width=10, 
                  head_length=10, fc='red', ec='red', label='Right')
    axes[1].arrow(center[0], center[1], -80, 0, head_width=10, 
                  head_length=10, fc='blue', ec='blue', label='Left')
    axes[1].arrow(center[0], center[1], 0, -80, head_width=10, 
                  head_length=10, fc='green', ec='green', label='Up')
    axes[1].arrow(center[0], center[1], 0, 80, head_width=10, 
                  head_length=10, fc='orange', ec='orange', label='Down')
    
    # Draw gaze direction
    gaze_x = -np.sin(np.radians(yaw)) * np.cos(np.radians(pitch)) * 60
    gaze_y = -np.sin(np.radians(pitch)) * 60
    
    axes[1].arrow(center[0], center[1], gaze_x, gaze_y, 
                  head_width=15, head_length=15, fc='black', ec='black', 
                  linewidth=3, label='Gaze Direction')
    
    axes[1].set_xlim(0, 200)
    axes[1].set_ylim(0, 200)
    axes[1].invert_yaxis()
    axes[1].set_title(f'Gaze Direction\nYaw: {yaw:.1f}°, Pitch: {pitch:.1f}°')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    
    plt.show()

def main():
    args = parse_args()
    
    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    
    # Load and preprocess image
    print(f"\n📷 Loading image: {args.image_path}")
    img = load_image(args.image_path)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
    
    # Load model
    print(f"\n🧠 Loading model: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"⚠️ Model not found. Creating new model...")
        model = PhysGaze()
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        model = PhysGaze()
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Run inference
    print(f"\n🔮 Running inference...")
    with torch.no_grad():
        gaze_pred = model.inference(img_tensor)
        yaw, pitch = gaze_pred[0].cpu().numpy()
    
    print(f"\n📊 Prediction Results:")
    print(f"  Yaw: {yaw:.2f}°")
    print(f"  Pitch: {pitch:.2f}°")
    
    # Visualize results
    print(f"\n🎨 Generating visualization...")
    visualize_gaze(img, yaw, pitch, save_path='gaze_visualization.png')
    
    print(f"\n✅ Inference completed!")

if __name__ == '__main__':
    main()
