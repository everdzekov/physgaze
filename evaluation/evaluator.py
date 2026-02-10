import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple

from ..models.physgaze import PhysGaze
from ..data.datasets import MPIIGazeLOSODataset
from ..training.trainer import EnhancedPhysGazeTrainer
from ..utils.visualization import EnhancedVisualizer
from .metrics import compute_gaze_metrics

class EnhancedLOSOEvaluator:
    """
    Enhanced LOSO evaluator with dynamic epoch stopping
    """
    
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self.all_subjects = sorted(df['subject'].unique())
        
        # Enhanced configuration
        self.config['early_stopping_patience'] = config.get('early_stopping_patience', 10)
        self.config['early_stopping_min_delta'] = config.get('early_stopping_min_delta', 0.001)
        self.config['overfitting_threshold'] = config.get('overfitting_threshold', 0.1)
        self.config['max_epochs'] = config.get('max_epochs', 100)
        
        print(f"\n🎯 ENHANCED LOSO EVALUATION SETUP")
        print(f"Total subjects: {len(self.all_subjects)}")
        print(f"Subjects: {self.all_subjects}")
        
        # Store results
        self.results = {}
        self.fold_histories = {}
        
        # Visualizer
        self.visualizer = EnhancedVisualizer()
    
    def run(self, device: torch.device) -> Dict:
        """Run enhanced LOSO cross-validation"""
        print("\n" + "="*80)
        print("🚀 ENHANCED LOSO CROSS-VALIDATION WITH DYNAMIC EPOCHS")
        print("="*80)
        
        fold_results = {}
        fold_training_dynamics = {}
        
        # Create LOSO folds
        for fold_idx, test_subject in enumerate(self.all_subjects):
            print(f"\n\n{'='*70}")
            print(f"🧪 FOLD {fold_idx + 1}/{len(self.all_subjects)}")
            print(f"Test Subject: {test_subject}")
            print(f"{'='*70}")
            
            # Create datasets
            remaining = [s for s in self.all_subjects if s != test_subject]
            n_val = max(1, int(len(remaining) * 0.2))
            val_subjects = remaining[:n_val]
            train_subjects = remaining[n_val:]
            
            print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
            print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
            
            train_dataset = MPIIGazeLOSODataset(
                self.df, test_subject, val_subjects, mode='train'
            )
            val_dataset = MPIIGazeLOSODataset(
                self.df, test_subject, val_subjects, mode='val'
            )
            test_dataset = MPIIGazeLOSODataset(
                self.df, test_subject, [], mode='test'
            )
            
            # Skip if datasets are empty
            if len(train_dataset) == 0:
                print(f"⚠️ No training data for fold {fold_idx}. Skipping.")
                continue
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2
            )
            
            # Train and evaluate this fold
            fold_result, fold_history = self._train_and_evaluate_fold(
                fold_idx, test_subject,
                train_loader, val_loader, test_loader,
                device
            )
            
            if fold_result:
                fold_results[test_subject] = fold_result
                fold_training_dynamics[test_subject] = fold_history
        
        # Calculate overall statistics
        if fold_results:
            self._calculate_enhanced_statistics(fold_results, fold_training_dynamics)
        
        return self.results
    
    def _train_and_evaluate_fold(self, fold_idx: int, test_subject: str,
                                train_loader: DataLoader, val_loader: DataLoader,
                                test_loader: DataLoader, device: torch.device):
        """Train and evaluate one fold"""
        # Create model
        model = PhysGaze(image_size=(36, 60))
        
        # Create trainer
        trainer = EnhancedPhysGazeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=self.config
        )
        
        # Train with dynamic epochs
        print(f"\n🏋️ Training for fold {fold_idx + 1} (Test: {test_subject})...")
        history = trainer.train(max_epochs=self.config['max_epochs'])
        
        # Store training history
        self.fold_histories[test_subject] = history
        
        # Evaluate on test set
        print(f"\n🧪 Evaluating on test subject {test_subject}...")
        test_results = self._evaluate_model(model, test_loader, test_subject, device)
        
        if test_results:
            print(f"\n📊 Results for {test_subject}:")
            print(f"  MAE: {test_results['mae']:.2f}°")
            print(f"  Yaw MAE: {test_results['mae_yaw']:.2f}°")
            print(f"  Pitch MAE: {test_results['mae_pitch']:.2f}°")
            print(f"  Init Outliers: {test_results['init_outlier_rate']:.1f}%")
            print(f"  Final Outliers: {test_results['final_outlier_rate']:.1f}%")
            print(f"  Outlier Reduction: {test_results['outlier_reduction']:.1f}%")
            print(f"  Samples: {test_results['n_samples']:,}")
            print(f"  Training Epochs: {len(history['train_loss'])}")
        
        return test_results, history
    
    def _evaluate_model(self, model: PhysGaze, test_loader: DataLoader,
                       test_subject: str, device: torch.device) -> Optional[Dict]:
        """Evaluate model on test set"""
        model.eval()
        
        all_preds = []
        all_targets = []
        all_inits = []
        init_outliers = []
        final_outliers = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f"Testing {test_subject}"):
                images, targets = images.to(device), targets.to(device)
                
                # Get predictions
                predictions = model(images, return_all=True)
                gaze_pred = predictions['gaze_corrected']
                gaze_init = predictions['gaze_init']
                
                # Calculate outliers
                init_outlier = model.get_outlier_rate(gaze_init)
                final_outlier = model.get_outlier_rate(gaze_pred)
                
                # Store results
                all_preds.append(gaze_pred.cpu())
                all_targets.append(targets.cpu())
                all_inits.append(gaze_init.cpu())
                init_outliers.append(init_outlier)
                final_outliers.append(final_outlier)
        
        if not all_preds:
            return None
        
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_inits = torch.cat(all_inits, dim=0)
        
        # Calculate metrics
        metrics = compute_gaze_metrics(all_preds, all_targets)
        
        init_outlier_rate = np.mean(init_outliers) * 100
        final_outlier_rate = np.mean(final_outliers) * 100
        
        results = {
            'mae': metrics['mae'],
            'mae_yaw': metrics['mae_yaw'],
            'mae_pitch': metrics['mae_pitch'],
            'rmse': metrics['rmse'],
            'mean_angular_error': metrics['mean_angular_error'],
            'std_angular_error': metrics['std_angular_error'],
            'init_outlier_rate': init_outlier_rate,
            'final_outlier_rate': final_outlier_rate,
            'outlier_reduction': ((init_outlier_rate - final_outlier_rate) /
                                 max(init_outlier_rate, 1e-10) * 100),
            'n_samples': len(all_preds),
            'predictions': all_preds.numpy(),
            'targets': all_targets.numpy(),
            'initial_predictions': all_inits.numpy(),
            'angular_errors': metrics['angular_errors'].numpy(),
            **{k: v for k, v in metrics.items() if k.startswith('acc_')}
        }
        
        # Store in results
        self.results[test_subject] = results
        
        return results
    
    def _calculate_enhanced_statistics(self, fold_results: Dict, 
                                      fold_training_dynamics: Dict):
        """Calculate enhanced statistics"""
        print("\n" + "="*80)
        print("📈 ENHANCED LOSO CROSS-VALIDATION FINAL RESULTS")
        print("="*80)
        
        # Extract metrics
        subjects = list(fold_results.keys())
        maes = [fold_results[s]['mae'] for s in subjects]
        maes_yaw = [fold_results[s]['mae_yaw'] for s in subjects]
        maes_pitch = [fold_results[s]['mae_pitch'] for s in subjects]
        init_outliers = [fold_results[s]['init_outlier_rate'] for s in subjects]
        final_outliers = [fold_results[s]['final_outlier_rate'] for s in subjects]
        outlier_reductions = [fold_results[s]['outlier_reduction'] for s in subjects]
        samples = [fold_results[s]['n_samples'] for s in subjects]
        training_epochs = [len(fold_training_dynamics[s]['train_loss']) for s in subjects]
        
        # Calculate statistics
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        mean_mae_yaw = np.mean(maes_yaw)
        std_mae_yaw = np.std(maes_yaw)
        mean_mae_pitch = np.mean(maes_pitch)
        std_mae_pitch = np.std(maes_pitch)
        
        mean_init_outliers = np.mean(init_outliers)
        mean_final_outliers = np.mean(final_outliers)
        mean_outlier_reduction = np.mean(outlier_reductions)
        mean_training_epochs = np.mean(training_epochs)
        
        # Print results
        print(f"\n📊 Overall Performance:")
        print(f"  Mean MAE: {mean_mae:.2f}° ± {std_mae:.2f}°")
        print(f"  Mean Yaw MAE: {mean_mae_yaw:.2f}° ± {std_mae_yaw:.2f}°")
        print(f"  Mean Pitch MAE: {mean_mae_pitch:.2f}° ± {std_mae_pitch:.2f}°")
        
        print(f"\n👁️ Outlier Analysis:")
        print(f"  Initial Outlier Rate: {mean_init_outliers:.1f}%")
        print(f"  Final Outlier Rate: {mean_final_outliers:.1f}%")
        print(f"  Mean Outlier Reduction: {mean_outlier_reduction:.1f}%")
        
        print(f"\n⏱️ Training Efficiency:")
        print(f"  Mean Training Epochs: {mean_training_epochs:.1f}")
        print(f"  Epoch Range: [{min(training_epochs)}, {max(training_epochs)}]")
        
        # Save results
        self._save_results(fold_results, fold_training_dynamics, 
                          mean_mae, std_mae, mean_training_epochs)
    
    def _save_results(self, fold_results: Dict, fold_training_dynamics: Dict,
                     mean_mae: float, std_mae: float, mean_training_epochs: float):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary DataFrame
        summary_data = []
        for subject in fold_results.keys():
            results = fold_results[subject]
            summary_data.append({
                'Subject': subject,
                'MAE (°)': f"{results['mae']:.2f}",
                'Yaw MAE (°)': f"{results['mae_yaw']:.2f}",
                'Pitch MAE (°)': f"{results['mae_pitch']:.2f}",
                'Init Outliers (%)': f"{results['init_outlier_rate']:.1f}",
                'Final Outliers (%)': f"{results['final_outlier_rate']:.1f}",
                'Outlier Reduction (%)': f"{results['outlier_reduction']:.1f}",
                'Training Epochs': len(fold_training_dynamics[subject]['train_loss']),
                'Samples': results['n_samples']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = f"results/physgaze_loso_results_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved results to: {csv_path}")
        
        # Save detailed results as JSON
        results_dict = {
            'timestamp': timestamp,
            'config': self.config,
            'overall_metrics': {
                'mean_mae': float(mean_mae),
                'std_mae': float(std_mae),
                'mean_training_epochs': float(mean_training_epochs),
                'n_subjects': len(fold_results),
                'total_samples': int(summary_df['Samples'].sum())
            },
            'per_subject_metrics': {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                                       for kk, vv in v.items() if kk not in ['predictions', 'targets', 'initial_predictions', 'angular_errors']}
                                   for k, v in fold_results.items()}
        }
        
        json_path = f"results/physgaze_loso_detailed_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"💾 Saved detailed results to: {json_path}")
