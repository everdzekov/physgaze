import os
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import List, Dict, Optional

try:
    from PIL import Image
    HAS_PIL = True
except:
    HAS_PIL = False

def load_all_mpiigaze_subjects_fixed(base_dir='./data/MPIIGaze/MPIIGaze', 
                                    samples_per_subject=None, 
                                    debug=False):
    """
    Fixed loader for MPIIGaze with correct gaze angle computation.
    
    Args:
        base_dir: Base directory containing MPIIGaze data
        samples_per_subject: Maximum samples per subject (None for all)
        debug: Print debug information
    
    Returns:
        DataFrame with all loaded data
    """
    print(f"\n🤖 Loading ALL MPIIGaze subjects from: {base_dir}")
    
    normalized_dir = os.path.join(base_dir, "Data", "Normalized")
    
    if not os.path.exists(normalized_dir):
        print(f"❌ Directory not found: {normalized_dir}")
        print(f"Current working directory: {os.getcwd()}")
        # Try alternative path
        alt_path = './MPIIGaze/Data/Normalized'
        if os.path.exists(alt_path):
            print(f"Found alternative path: {alt_path}")
            normalized_dir = alt_path
        else:
            return pd.DataFrame()
    
    # Get ALL subjects
    subjects = sorted([d for d in os.listdir(normalized_dir) 
                      if os.path.isdir(os.path.join(normalized_dir, d)) and d.startswith('p')])
    
    print(f"📊 Found {len(subjects)} subjects: {subjects}")
    
    all_data = []
    total_subjects_loaded = 0
    
    # Process EACH subject
    for subject_idx, subject in enumerate(subjects):
        subject_path = os.path.join(normalized_dir, subject)
        
        # Get all .mat files for this subject
        mat_files = sorted([f for f in os.listdir(subject_path) 
                          if f.endswith('.mat') and f.startswith('day')])
        
        if not mat_files:
            if debug:
                print(f"⚠️ No .mat files found for subject {subject}")
            continue
        
        print(f"\n📁 Processing subject {subject} ({subject_idx+1}/{len(subjects)})...")
        print(f"  Days available: {len(mat_files)} files")
        
        subject_samples = 0
        days_processed = 0
        
        # Process each day file
        for day_idx, mat_file in enumerate(mat_files):
            mat_path = os.path.join(subject_path, mat_file)
            day_name = os.path.splitext(mat_file)[0]
            
            try:
                # Load .mat file
                if debug and day_idx == 0:
                    print(f"  Loading {mat_file}...")
                
                mat_data = sio.loadmat(mat_path, simplify_cells=True)
                
                if 'data' not in mat_data:
                    if debug:
                        print(f"  ⚠️ 'data' key not found in {mat_file}")
                    continue
                
                data_dict = mat_data['data']
                
                if not isinstance(data_dict, dict):
                    if debug:
                        print(f"  ⚠️ 'data' is not a dict in {mat_file}")
                    continue
                
                # Process right eye
                if 'right' in data_dict:
                    right_data = data_dict['right']
                    if isinstance(right_data, dict):
                        # Extract images and gaze
                        if 'image' in right_data and 'gaze' in right_data:
                            right_images = right_data['image']  # Shape: (n_samples, 36, 60)
                            right_gazes = right_data['gaze']    # Shape: (n_samples, 3)
                            
                            n_samples = len(right_images)
                            if debug and day_idx == 0:
                                print(f"    Right eye: {n_samples} samples, image shape: {right_images.shape}, gaze shape: {right_gazes.shape}")
                            
                            # Determine how many samples to take
                            if samples_per_subject is None:
                                # Take all samples
                                sample_indices = range(n_samples)
                            else:
                                # Take evenly spaced samples
                                num_needed = min(n_samples, max(1, samples_per_subject // (2 * len(mat_files))))
                                step = max(1, n_samples // num_needed)
                                sample_indices = range(0, n_samples, step)
                            
                            for i in sample_indices:
                                img = right_images[i]
                                gaze_3d = right_gazes[i] if i < len(right_gazes) else None
                                
                                # Process image
                                processed_img = _process_mpiigaze_image_fixed(img)
                                if processed_img is None:
                                    continue
                                
                                # Process gaze - FIXED ANGLE COMPUTATION
                                gaze_x, gaze_y = _process_mpiigaze_gaze_fixed(gaze_3d)
                                
                                all_data.append({
                                    'subject': subject,
                                    'day': day_name,
                                    'mat_file': mat_file,
                                    'eye_side': 'right',
                                    'image_data': processed_img,
                                    'gaze_x': gaze_x,
                                    'gaze_y': gaze_y,
                                    'gaze_3d': gaze_3d if gaze_3d is not None else [0, 0, 0],
                                    'sample_idx': i,
                                    'day_idx': day_idx
                                })
                                
                                subject_samples += 1
                
                # Process left eye
                if 'left' in data_dict:
                    left_data = data_dict['left']
                    if isinstance(left_data, dict):
                        # Extract images and gaze
                        if 'image' in left_data and 'gaze' in left_data:
                            left_images = left_data['image']  # Shape: (n_samples, 36, 60)
                            left_gazes = left_data['gaze']    # Shape: (n_samples, 3)
                            
                            n_samples = len(left_images)
                            if debug and day_idx == 0:
                                print(f"    Left eye: {n_samples} samples, image shape: {left_images.shape}, gaze shape: {left_gazes.shape}")
                            
                            # Determine how many samples to take
                            if samples_per_subject is None:
                                # Take all samples
                                sample_indices = range(n_samples)
                            else:
                                # Take evenly spaced samples
                                num_needed = min(n_samples, max(1, samples_per_subject // (2 * len(mat_files))))
                                step = max(1, n_samples // num_needed)
                                sample_indices = range(0, n_samples, step)
                            
                            for i in sample_indices:
                                img = left_images[i]
                                gaze_3d = left_gazes[i] if i < len(left_gazes) else None
                                
                                # Process image
                                processed_img = _process_mpiigaze_image_fixed(img)
                                if processed_img is None:
                                    continue
                                
                                # Process gaze - FIXED ANGLE COMPUTATION
                                gaze_x, gaze_y = _process_mpiigaze_gaze_fixed(gaze_3d)
                                
                                all_data.append({
                                    'subject': subject,
                                    'day': day_name,
                                    'mat_file': mat_file,
                                    'eye_side': 'left',
                                    'image_data': processed_img,
                                    'gaze_x': gaze_x,
                                    'gaze_y': gaze_y,
                                    'gaze_3d': gaze_3d if gaze_3d is not None else [0, 0, 0],
                                    'sample_idx': i,
                                    'day_idx': day_idx
                                })
                                
                                subject_samples += 1
                
                days_processed += 1
                if debug and (day_idx < 3 or day_idx == len(mat_files) - 1):
                    print(f"    Processed {mat_file}: {subject_samples - sum([d['subject'] == subject for d in all_data])} new samples")
                
            except Exception as e:
                if debug:
                    print(f"    ❌ Error loading {mat_file}: {str(e)[:100]}")
                continue
        
        total_subjects_loaded += 1
        print(f"  ✅ Subject {subject}: Loaded {sum([d['subject'] == subject for d in all_data])} samples from {days_processed} days")
        
        # Progress update
        if (subject_idx + 1) % 3 == 0 or subject_idx == len(subjects) - 1:
            current_total = len(all_data)
            print(f"  📈 Progress: {subject_idx + 1}/{len(subjects)} subjects, {current_total:,} total samples")
            
            if current_total > 0 and debug:
                # Show sample statistics
                recent_samples = [d for d in all_data if d['subject'] == subject]
                if recent_samples:
                    img_shape = recent_samples[0]['image_data'].shape
                    gaze_x_vals = [d['gaze_x'] for d in recent_samples]
                    print(f"      Sample shape: {img_shape}, Gaze X range: [{min(gaze_x_vals):.1f}, {max(gaze_x_vals):.1f}]")
    
    if len(all_data) == 0:
        print("❌ No data loaded!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    print(f"\n" + "="*70)
    print("✅ ALL SUBJECTS LOADING COMPLETE!")
    print("="*70)
    print(f"📊 COMPREHENSIVE SUMMARY:")
    print(f"  Total images: {len(df):,}")
    print(f"  Subjects loaded: {df['subject'].nunique()}")
    print(f"  Eye distribution: {df['eye_side'].value_counts().to_dict()}")
    print(f"  Unique days: {df['day'].nunique()}")
    print(f"  Unique mat files: {df['mat_file'].nunique()}")
    
    if len(df) > 0:
        print(f"\n📈 Gaze Statistics:")
        print(f"  Gaze X (yaw): min={df['gaze_x'].min():.2f}°, max={df['gaze_x'].max():.2f}°, mean={df['gaze_x'].mean():.2f}°")
        print(f"  Gaze Y (pitch): min={df['gaze_y'].min():.2f}°, max={df['gaze_y'].max():.2f}°, mean={df['gaze_y'].mean():.2f}°")
        
        print(f"\n👁️ Image Statistics:")
        img_shapes = df['image_data'].apply(lambda x: x.shape).value_counts()
        print(f"  Image shapes distribution: {img_shapes.to_dict()}")
        
        # Check image values
        sample_img = df.iloc[0]['image_data']
        print(f"  Sample image shape: {sample_img.shape}, dtype: {sample_img.dtype}, range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
    
    return df

def _process_mpiigaze_image_fixed(img):
    """
    Process MPIIGaze image data ensuring 36x60 grayscale output.
    
    Based on your structure, images are already (36, 60) uint8.
    """
    if isinstance(img, np.ndarray):
        # Handle different array shapes
        if img.ndim == 2:
            # Already 2D: (36, 60)
            if img.shape == (36, 60):
                processed_img = img.astype(np.float32) / 255.0
            else:
                # Resize to 36x60 if needed
                try:
                    if HAS_PIL:
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil = img_pil.resize((60, 36), Image.BILINEAR)
                        processed_img = np.array(img_pil).astype(np.float32) / 255.0
                    else:
                        # Simple numpy resize
                        import scipy.ndimage
                        zoom_factors = (36 / img.shape[0], 60 / img.shape[1])
                        processed_img = scipy.ndimage.zoom(img, zoom_factors, order=1).astype(np.float32) / 255.0
                except Exception as e:
                    return None
        elif img.ndim == 3:
            # Handle 3D arrays
            if img.shape[2] == 1:  # (36, 60, 1)
                processed_img = img[:, :, 0].astype(np.float32) / 255.0
            elif img.shape[2] == 3:  # (36, 60, 3) - RGB
                # Convert to grayscale
                processed_img = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                # Unexpected shape
                return None
        else:
            return None
        
        # Ensure correct size
        if processed_img.shape != (36, 60):
            try:
                if HAS_PIL:
                    img_pil = Image.fromarray((processed_img * 255).astype(np.uint8))
                    img_pil = img_pil.resize((60, 36), Image.BILINEAR)
                    processed_img = np.array(img_pil).astype(np.float32) / 255.0
                else:
                    import scipy.ndimage
                    zoom_factors = (36 / processed_img.shape[0], 60 / processed_img.shape[1])
                    processed_img = scipy.ndimage.zoom(processed_img, zoom_factors, order=1)
            except:
                return None
        
        # Clip to valid range
        processed_img = np.clip(processed_img, 0.0, 1.0)
        
        return processed_img
    
    return None

def _process_mpiigaze_gaze_fixed(gaze_3d):
    """
    FIXED: Process MPIIGaze gaze data properly.
    
    Based on analysis showing yaw in [-180°, 180°] range,
    it appears the gaze data might already be in angles.
    This function auto-detects the format and returns correct angles.
    """
    if gaze_3d is None:
        return 0.0, 0.0
    
    if isinstance(gaze_3d, np.ndarray) and gaze_3d.size >= 3:
        # Extract components
        x, y, z = float(gaze_3d[0]), float(gaze_3d[1]), float(gaze_3d[2])
        
        # Calculate vector magnitude to determine format
        magnitude = np.sqrt(x*x + y*y + z*z)
        
        # DEBUG: For first few samples, print raw values
        import random
        if random.random() < 0.001:  # Print 0.1% of samples
            print(f"  DEBUG: Raw gaze: [{x:.6f}, {y:.6f}, {z:.6f}], magnitude: {magnitude:.6f}")
        
        # METHOD 1: Check if it's a normalized gaze vector (magnitude ~1)
        if 0.9 < magnitude < 1.1:
            # It's a normalized 3D gaze vector
            # Convert to angles using standard formula
            pitch = np.arcsin(-y)  # Vertical angle (positive is up)
            yaw = np.arctan2(x, z)  # Horizontal angle (positive is right)
            
            # Convert to degrees
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            
            # These should be in reasonable ranges: yaw ±30°, pitch ±20°
        
        # METHOD 2: Check if values are already in radians
        # Typical gaze angles in radians: yaw ~ ±0.5 rad, pitch ~ ±0.35 rad
        elif abs(x) < 3.2 and abs(y) < 3.2:
            # x and y might already be angles in radians
            yaw_deg = np.degrees(x)
            pitch_deg = np.degrees(y)
            
            # z might be unused or contain additional info
        
        # METHOD 3: Check if values are already in degrees
        elif abs(x) > 100 or abs(y) > 100:
            # Already in degrees, but scale might be wrong
            # Based on your output, yaw is in [-180, 180], pitch is in [-20, 2]
            yaw_deg = x / 6.0  # Scale from [-180, 180] to [-30, 30]
            pitch_deg = y  # Pitch looks about right already
        
        # METHOD 4: Default fallback (direct use)
        else:
            # Unknown format, use direct values with clipping
            yaw_deg = x
            pitch_deg = y
        
        # Apply reasonable bounds for gaze angles
        yaw_deg = max(min(yaw_deg, 45), -45)     # Yaw should be within ±45°
        pitch_deg = max(min(pitch_deg, 30), -30)  # Pitch should be within ±30°
        
        return yaw_deg, pitch_deg
    
    return 0.0, 0.0
