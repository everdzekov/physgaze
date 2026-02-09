import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_samples(dataset, num_samples=5):
    """Visualize random samples from dataset."""
    print(f"\n👁️ Visualizing {num_samples} samples...")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        image, gaze = dataset[idx]
        
        # Denormalize gaze
        gaze_deg = gaze.numpy() * np.array([30.0, 20.0])
        
        # Display image
        img_display = image.squeeze().numpy()
        axes[i].imshow(img_display, cmap='gray')
        axes[i].set_title(f"Gaze: ({gaze_deg[0]:.1f}°, {gaze_deg[1]:.1f}°)")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history['train_mae'], label='Train')
    axes[1].plot(history['val_mae'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (°)')
    axes[1].set_title('Gaze Estimation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(history['learning_rate'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
