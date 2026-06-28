"""
PHASE 2: AUTOMATED ABLATION STUDIES
====================================
Runs controlled experiments with different feature/augmentation combinations.
Each experiment trains on the same stratified split with minimal epochs for speed.

Usage:
    python ablation_studies.py --fast  # Fast approximation (10 epochs)
    python ablation_studies.py --full  # Full training (60 epochs)

Results saved to: ablation_results.json and ablation_summary.csv
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple
import csv
from datetime import datetime

# Import project modules
from config import get_config, TrainingConfig, FrameFeaturesConfig, SpatialFeaturesConfig
from src.training.train import create_data_loaders, train_one_epoch, FocalLoss, _compute_inverse_class_weights
from src.training.model import SignLanguageGRU
from src.preprocessing.dataset import ISLDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")


@dataclass
class AblationExperiment:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    # Augmentation flags
    augmentation_enabled: bool = True
    mixup_enabled: bool = True
    class_weighting_enabled: bool = True
    label_smoothing: float = 0.05
    # Feature flags
    use_velocity: bool = True
    use_face_relative: bool = True
    # Model flags
    use_attention: bool = True
    use_proximity_bias: bool = True
    bidirectional: bool = True
    
    # Training params
    num_epochs: int = 10  # Fast ablation


class AblationRunner:
    def __init__(self, output_dir: str = "ablation_output", fast_mode: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fast_mode = fast_mode
        self.results: List[Dict] = []
        
    def define_experiments(self) -> List[AblationExperiment]:
        """Define all ablation experiments to run."""
        return [
            # ===== BASELINE =====
            AblationExperiment(
                name="BASELINE_Full",
                description="Full pipeline with all features and augmentation",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=20 if not self.fast_mode else 10,
            ),
            
            # ===== FEATURE ABLATIONS =====
            AblationExperiment(
                name="Feature_NoVelocity",
                description="Without velocity features (only position)",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                use_velocity=False,  # ABLATE
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            AblationExperiment(
                name="Feature_NoFaceRelative",
                description="Without face-relative normalization (only raw coordinates)",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=False,  # ABLATE
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            
            # ===== AUGMENTATION ABLATIONS =====
            AblationExperiment(
                name="Augment_None",
                description="Baseline: no augmentation",
                augmentation_enabled=False,  # ABLATE
                mixup_enabled=False,  # ABLATE
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            AblationExperiment(
                name="Augment_NoMixup",
                description="With online augmentation but no mixup",
                augmentation_enabled=True,
                mixup_enabled=False,  # ABLATE
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            AblationExperiment(
                name="Augment_NoWeighting",
                description="With augmentation but no class weighting",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=False,  # ABLATE
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            AblationExperiment(
                name="Augment_NoSmoothing",
                description="With augmentation but no label smoothing",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                label_smoothing=0.0,  # ABLATE
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            
            # ===== ARCHITECTURE ABLATIONS =====
            AblationExperiment(
                name="Model_NoAttention",
                description="Without hybrid attention layer",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=True,
                use_attention=False,  # ABLATE
                use_proximity_bias=True,
                bidirectional=True,
                num_epochs=10,
            ),
            AblationExperiment(
                name="Model_NoBidirectional",
                description="Unidirectional GRU instead of bidirectional",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=True,
                bidirectional=False,  # ABLATE
                num_epochs=10,
            ),
            AblationExperiment(
                name="Model_NoProximityBias",
                description="Attention without proximity biasing",
                augmentation_enabled=True,
                mixup_enabled=True,
                class_weighting_enabled=True,
                use_velocity=True,
                use_face_relative=True,
                use_attention=True,
                use_proximity_bias=False,  # ABLATE
                bidirectional=True,
                num_epochs=10,
            ),
        ]
    
    def run_experiment(self, exp: AblationExperiment) -> Dict:
        """Run a single ablation experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {exp.name}")
        logger.info(f"Description: {exp.description}")
        logger.info(f"{'='*70}")
        
        # Load configuration
        cfg = get_config()
        
        # Override with ablation settings
        cfg.frame_features.use_velocity = exp.use_velocity
        cfg.spatial.use_face_relative = exp.use_face_relative
        cfg.model.use_face_proximity_attention = exp.use_proximity_bias
        cfg.model.bidirectional = exp.bidirectional
        cfg.training.use_mixup = exp.mixup_enabled
        cfg.training.use_class_weights = exp.class_weighting_enabled
        cfg.training.label_smoothing = exp.label_smoothing
        cfg.training.num_epochs = exp.num_epochs
        
        try:
            # Create data loaders
            train_loader, val_loader, num_classes, class_weights, full_ds = create_data_loaders()
            
            # Create model
            input_size = cfg.frame_features.input_sequence_dim
            model = SignLanguageGRU(
                input_size=input_size,
                num_classes=num_classes,
                use_attention=exp.use_attention,
            ).to(DEVICE)
            
            # Setup loss and optimizer
            if cfg.training.use_mixup:
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights if exp.class_weighting_enabled else None,
                    label_smoothing=exp.label_smoothing,
                    reduction='mean'
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights if exp.class_weighting_enabled else None,
                    label_smoothing=exp.label_smoothing,
                    reduction='mean'
                )
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=cfg.training.lr_min
            )
            
            # Train for specified epochs
            best_val_acc = 0.0
            best_epoch = 0
            
            for epoch in range(exp.num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                for seq, proximity, labels in train_loader:
                    seq = seq.to(DEVICE)
                    proximity = proximity.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    
                    if exp.use_attention:
                        logits = model(seq, proximity=proximity)
                    else:
                        logits = model(seq, proximity=None)
                    
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                # Validation
                model.eval()
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for seq, proximity, labels in val_loader:
                        seq = seq.to(DEVICE)
                        proximity = proximity.to(DEVICE)
                        labels = labels.to(DEVICE)
                        
                        if exp.use_attention:
                            logits = model(seq, proximity=proximity)
                        else:
                            logits = model(seq, proximity=None)
                        
                        preds = logits.argmax(dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(labels.cpu().numpy())
                
                val_acc = accuracy_score(val_labels, val_preds)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                
                if (epoch + 1) % max(1, exp.num_epochs // 3) == 0 or epoch == 0:
                    logger.info(f"  Epoch {epoch+1}/{exp.num_epochs}: Val Acc = {val_acc:.4f}")
            
            # Final metrics
            prec, rec, f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average='macro', zero_division=0
            )
            
            result = {
                'experiment': exp.name,
                'description': exp.description,
                'accuracy': float(best_val_acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'best_epoch': best_epoch,
                'num_epochs': exp.num_epochs,
                'timestamp': datetime.now().isoformat(),
                'config_summary': {
                    'use_velocity': exp.use_velocity,
                    'use_face_relative': exp.use_face_relative,
                    'use_attention': exp.use_attention,
                    'use_proximity_bias': exp.use_proximity_bias,
                    'bidirectional': exp.bidirectional,
                    'augmentation_enabled': exp.augmentation_enabled,
                    'mixup_enabled': exp.mixup_enabled,
                    'class_weighting_enabled': exp.class_weighting_enabled,
                }
            }
            
            logger.info(f"✓ {exp.name}: Accuracy={result['accuracy']:.4f}, "
                       f"Precision={result['precision']:.4f}, "
                       f"Recall={result['recall']:.4f}, F1={result['f1']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ {exp.name} failed: {str(e)}")
            return {
                'experiment': exp.name,
                'description': exp.description,
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
    
    def run_all(self) -> None:
        """Run all ablation experiments."""
        experiments = self.define_experiments()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 2: ABLATION STUDIES")
        logger.info(f"Total experiments: {len(experiments)}")
        logger.info(f"Mode: {'FAST (10 epochs)' if self.fast_mode else 'FULL (20+ epochs)'}")
        logger.info(f"{'='*70}\n")
        
        for i, exp in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{len(experiments)}]")
            result = self.run_experiment(exp)
            self.results.append(result)
        
        # Save results
        self.save_results()
    
    def save_results(self) -> None:
        """Save ablation results to JSON and CSV."""
        # JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n✓ Results saved to {json_path}")
        
        # CSV
        csv_path = self.output_dir / "ablation_summary.csv"
        if self.results:
            fieldnames = ['experiment', 'description', 'accuracy', 'precision', 'recall', 'f1']
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = {k: result.get(k, '') for k in fieldnames}
                    writer.writerow(row)
            logger.info(f"✓ Summary saved to {csv_path}")
        
        # Print summary table
        logger.info(f"\n{'='*90}")
        logger.info(f"{'ABLATION SUMMARY':^90}")
        logger.info(f"{'='*90}")
        logger.info(f"{'Experiment':<30} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
        logger.info(f"{'-'*90}")
        for result in sorted(self.results, key=lambda x: x.get('accuracy', 0), reverse=True):
            acc = result.get('accuracy', 0)
            prec = result.get('precision', 0)
            rec = result.get('recall', 0)
            f1 = result.get('f1', 0)
            logger.info(f"{result['experiment']:<30} {acc:>11.2%} {prec:>11.2%} {rec:>11.2%} {f1:>11.2%}")
        logger.info(f"{'='*90}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run fast ablations (10 epochs)")
    parser.add_argument("--full", action="store_true", help="Run full ablations (20+ epochs)")
    parser.add_argument("--output", default="ablation_output", help="Output directory")
    
    args = parser.parse_args()
    fast_mode = args.fast or not args.full
    
    runner = AblationRunner(output_dir=args.output, fast_mode=fast_mode)
    runner.run_all()
