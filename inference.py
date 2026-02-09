import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gaze_networks import SimpleGazeNet

def preprocess_image(image_path, target_size=(60, 36)):
    """Preprocess image for gaze estimation."""
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize
    img = img.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Add channel dimension
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def predict_gaze(model, image_tensor, device='cpu'):
    """Predict gaze from image tensor."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
    
    # Denormalize output
    gaze_angles = output.cpu().numpy().flatten()
    gaze_angles[0] *= 30.0  # yaw
    gaze_angles[1] *= 20.0  # pitch
    
    return gaze_angles

def visualize_gaze(image_path, gaze_angles, save_path=None):
    """Visualize image with gaze direction."""
    img = Image.open(image_path).convert('RGB')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    
    # Create gaze visualization (simple arrow)
    yaw, pitch = gaze_angles
    arrow_length = 100
    dx = arrow_length * np.sin(np.radians(yaw))
    dy = -arrow_length * np.sin(np.radians(pitch))
    
    # Start arrow from center
    center_x, center_y = img.width // 2, img.height // 2
    ax.arrow(center_x, center_y, dx, dy, head_width=20, head_length=20, 
             fc='red', ec='red', linewidth=3)
    
    ax.set_title(f'Gaze: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°', fontsize=14)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()

def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gaze estimation inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_mpiigaze_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--visualize', action='store_true', help='Visualize result')
    parser.add_argument('--save', type=str, default=None, help='Save visualization path')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleGazeNet()
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        print(f"Model loaded from {args.model}")
    else:
        print(f"Model not found at {args.model}")
        return
    
    # Preprocess image
    if not os.path.exists(args.image):
        print(f"Image not found at {args.image}")
        return
    
    image_tensor = preprocess_image(args.image)
    
    # Predict gaze
    gaze_angles = predict_gaze(model, image_tensor, device)
    print(f"Predicted gaze angles: Yaw={gaze_angles[0]:.2f}°, Pitch={gaze_angles[1]:.2f}°")
    
    # Visualize if requested
    if args.visualize:
        visualize_gaze(args.image, gaze_angles, args.save)

if __name__ == "__main__":
    main()
