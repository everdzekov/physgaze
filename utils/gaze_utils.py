import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mpiigaze_loader import load_all_mpiigaze_subjects_fixed

def debug_gaze_values(base_dir='./data/MPIIGaze/MPIIGaze'):
    """Debug function to check raw gaze values vs computed angles."""
    print("🔍 DEBUGGING GAZE VALUES")
    print("=" * 70)
    
    # Load a small sample with debug mode
    df = load_all_mpiigaze_subjects_fixed(
        base_dir=base_dir,
        samples_per_subject=20,
        debug=False
    )
    
    if len(df) > 0:
        print(f"\n📊 Checking first 10 samples:")
        for i in range(min(10, len(df))):
            gaze_3d = df.iloc[i]['gaze_3d']
            yaw_computed = df.iloc[i]['gaze_x']
            pitch_computed = df.iloc[i]['gaze_y']
            
            print(f"\nSample {i}:")
            print(f"  Raw 3D gaze: [{gaze_3d[0]:.6f}, {gaze_3d[1]:.6f}, {gaze_3d[2]:.6f}]")
            print(f"  Magnitude: {np.sqrt(gaze_3d[0]**2 + gaze_3d[1]**2 + gaze_3d[2]**2):.6f}")
            print(f"  Computed angles: Yaw={yaw_computed:.2f}°, Pitch={pitch_computed:.2f}°")
            
            # Also compute using different methods for comparison
            x, y, z = gaze_3d[0], gaze_3d[1], gaze_3d[2]
            
            # Method 1: Direct as angles
            print(f"  As direct values: X={x:.6f}, Y={y:.6f}, Z={z:.6f}")
            
            # Method 2: As radians (if that's what they are)
            if abs(x) < 3.2:
                print(f"  As radians (converted): Yaw={np.degrees(x):.2f}°, Pitch={np.degrees(y):.2f}°")
        
        # Plot distribution of gaze angles
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['gaze_x'], bins=50, alpha=0.7)
        plt.xlabel('Yaw (degrees)')
        plt.ylabel('Frequency')
        plt.title('Yaw Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(df['gaze_y'], bins=50, alpha=0.7)
        plt.xlabel('Pitch (degrees)')
        plt.ylabel('Frequency')
        plt.title('Pitch Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.scatter(df['gaze_x'], df['gaze_y'], alpha=0.5, s=1)
        plt.xlabel('Yaw (degrees)')
        plt.ylabel('Pitch (degrees)')
        plt.title('Gaze Scatter Plot')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 Statistics after processing:")
        print(f"  Yaw range: [{df['gaze_x'].min():.2f}°, {df['gaze_x'].max():.2f}°]")
        print(f"  Pitch range: [{df['gaze_y'].min():.2f}°, {df['gaze_y'].max():.2f}°]")
        print(f"  Mean yaw: {df['gaze_x'].mean():.2f}°, Mean pitch: {df['gaze_y'].mean():.2f}°")
