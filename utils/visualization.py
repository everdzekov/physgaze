import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_results(
    preds_deg: np.ndarray,
    targets_deg: np.ndarray,
    angular_errors: np.ndarray,
    save_path: Optional[str] = None
):
    """Generate comprehensive visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Scatter plot of predictions vs targets
    ax = axes[0, 0]
    ax.scatter(targets_deg[:, 0], preds_deg[:, 0], alpha=0.3, s=5, c='#0ff4c6', label='Yaw')
    ax.scatter(targets_deg[:, 1], preds_deg[:, 1], alpha=0.3, s=5, c='#ff6b9d', label='Pitch')
    ax.plot([-60, 60], [-60, 60], 'k--', linewidth=1)
    ax.set_xlabel('Ground Truth (°)')
    ax.set_ylabel('Prediction (°)')
    ax.set_title('Predictions vs Ground Truth')
    ax.legend()
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(angular_errors, bins=50, color='#a855f7', edgecolor='white', alpha=0.7)
    ax.axvline(np.mean(angular_errors), color='#0ff4c6', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(angular_errors):.2f}°')
    ax.axvline(np.median(angular_errors), color='#ff6b9d', linestyle='--',
               linewidth=2, label=f'Median: {np.median(angular_errors):.2f}°')
    ax.set_xlabel('Angular Error (°)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Gaze distribution with constraint zone
    ax = axes[0, 2]
    theta = np.linspace(0, 2*np.pi, 100)
    constraint_x = 55 * np.cos(theta)
    constraint_y = 40 * np.sin(theta)
    ax.fill(constraint_x, constraint_y, color='#0ff4c6', alpha=0.1, label='Valid Region')
    ax.plot(constraint_x, constraint_y, color='#0ff4c6', linestyle='--', linewidth=2)
    ax.scatter(preds_deg[:, 0], preds_deg[:, 1], alpha=0.3, s=5, c='#a855f7')
    ax.set_xlabel('Yaw (°)')
    ax.set_ylabel('Pitch (°)')
    ax.set_title('Gaze Distribution with Constraints')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Error by gaze angle
    ax = axes[1, 0]
    gaze_magnitude = np.sqrt(targets_deg[:, 0]**2 + targets_deg[:, 1]**2)
    bins = [0, 10, 20, 30, 40, 50, 60]
    bin_errors = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = (gaze_magnitude >= bins[i]) & (gaze_magnitude < bins[i+1])
        if np.any(mask):
            bin_errors.append(np.mean(angular_errors[mask]))
            bin_centers.append((bins[i] + bins[i+1]) / 2)
    ax.bar(bin_centers, bin_errors, width=8, color='#0ff4c6', edgecolor='white')
    ax.set_xlabel('Gaze Magnitude (°)')
    ax.set_ylabel('Mean Error (°)')
    ax.set_title('Error vs Gaze Angle')
    ax.grid(True, alpha=0.3)
    
    # 5. Error heatmap
    ax = axes[1, 1]
    heatmap, xedges, yedges = np.histogram2d(
        targets_deg[:, 0], targets_deg[:, 1],
        bins=20, range=[[-50, 50], [-40, 30]],
        weights=angular_errors
    )
    counts, _, _ = np.histogram2d(
        targets_deg[:, 0], targets_deg[:, 1],
        bins=20, range=[[-50, 50], [-40, 30]]
    )
    heatmap = np.divide(heatmap, counts, where=counts > 0)
    heatmap[counts == 0] = np.nan
    im = ax.imshow(heatmap.T, origin='lower', extent=[-50, 50, -40, 30],
                   aspect='auto', cmap='magma')
    plt.colorbar(im, ax=ax, label='Mean Error (°)')
    ax.set_xlabel('Yaw (°)')
    ax.set_ylabel('Pitch (°)')
    ax.set_title('Error Heatmap')
    
    # 6. Cumulative error distribution
    ax = axes[1, 2]
    sorted_errors = np.sort(angular_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, color='#a855f7', linewidth=2)
    for thresh in [5, 10, 15]:
        pct = np.mean(angular_errors <= thresh) * 100
        ax.axvline(thresh, color='#64748b', linestyle=':', alpha=0.5)
        ax.text(thresh + 0.5, pct + 2, f'{pct:.1f}%', fontsize=9)
    ax.set_xlabel('Angular Error (°)')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Cumulative Error Distribution')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()