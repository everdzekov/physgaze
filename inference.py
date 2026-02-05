#!/usr/bin/env python3
"""
Inference script for single images.
"""

import torch
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models.physgaze import PhysGaze
from utils.helpers import denormalize_gaze


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with PhysGaze')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./output.png', help='Output image path')
    return parser.parse_args()


def preprocess_image(image_path: str, image_size=(36, 60)):
    """Preprocess input image for model."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((image_size[1], image_size[0]))  # (w, h) order
    
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Convert to tensor [1, 1, H, W]
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def visualize_gaze(image: np.ndarray, gaze: torch.Tensor):
    """Visualize gaze direction on image."""
    yaw = gaze[0].item() * 55.0  # Convert to degrees
    pitch = gaze[1].item() * 40.0
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input Eye Image')
    ax[0].axis('off')
    
    # Gaze visualization
    ax[1].set_xlim(-60, 60)
    ax[1].set_ylim(-45, 45)
    ax[1].set_xlabel('Yaw (°)')
    ax[1].set_ylabel('Pitch (°)')
    ax[1].set_title(f'Gaze Direction: ({yaw:.1f}°, {pitch:.1f}°)')
    ax[1].grid(True, alpha=0.3)
    
    # Draw constraint ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = 55 * np.cos(theta)
    ellipse_y = 40 * np.sin(theta)
    ax[1].plot(ellipse_x, ellipse_y, '--', color='#0ff4c6', alpha=0.5, label='Valid Region')
    ax[1].fill(ellipse_x, ellipse_y, color='#0ff4c6', alpha=0.1)
    
    # Plot gaze point
    ax[1].plot(yaw, pitch, 'ro', markersize=10, label='Predicted Gaze')
    ax[1].legend()
    
    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = PhysGaze(
        pretrained_backbone=False,
        use_acm=True,
        use_renderer=True,
        image_size=(36, 60)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Preprocess image
    image_tensor = preprocess_image(args.image).to(device)
    
    # Run inference
    with torch.no_grad():
        gaze_pred = model.predict(image_tensor)
    
    # Convert to degrees
    gaze_deg = denormalize_gaze(gaze_pred)
    
    print(f"\nPredicted gaze direction:")
    print(f"Yaw: {gaze_deg[0, 0].item():.2f}°")
    print(f"Pitch: {gaze_deg[0, 1].item():.2f}°")
    
    # Load original image for visualization
    original_image = np.array(Image.open(args.image).convert('L'))
    
    # Create visualization
    fig = visualize_gaze(original_image, gaze_pred[0])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {args.output}")
    plt.show()


if __name__ == '__main__':
    main()