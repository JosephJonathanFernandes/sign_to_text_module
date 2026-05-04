"""
Adapter training utilities for safe, asynchronous live adaptation.

This module:
- Manages threaded adapter training without blocking webcam FPS
- Enforces safety checks (class balance, performance validation)
- Handles weight backup/restore
- Logs adapter updates
"""

import threading
import numpy as np
import torch
import os
from typing import Optional, Dict, Any, Callable
from datetime import datetime


class AdapterTrainingManager:
    """Manage safe, threaded adapter training."""
    
    def __init__(
        self,
        adapter_trainer,
        num_classes: int,
        device: str = "cpu",
        enable_adaptation: bool = True,
        adapter_weights_dir: str = "adapter_weights/",
    ):
        """
        Args:
            adapter_trainer: AdapterTrainer instance
            num_classes: Number of classes
            device: torch device
            enable_adaptation: Enable/disable adapter training
            adapter_weights_dir: Directory to save adapter weights
        """
        self.adapter_trainer = adapter_trainer
        self.num_classes = num_classes
        self.device = device
        self.enable_adaptation = enable_adaptation
        self.adapter_weights_dir = adapter_weights_dir
        
        os.makedirs(adapter_weights_dir, exist_ok=True)
        
        # Threading
        self.training_thread: Optional[threading.Thread] = None
        self.training_lock = threading.Lock()
        self.stop_training = False
        
        # State
        self.is_training = False
        self.last_backup = None
        self.performance_log = []
        
        print(f"[AdapterManager] Initialized:")
        print(f"  Adaptation enabled: {enable_adaptation}")
        print(f"  Weights dir: {adapter_weights_dir}")
    
    def _backup_weights(self) -> dict:
        """Backup current adapter weights."""
        backup = self.adapter_trainer.model.get_checkpoint()
        self.last_backup = backup
        return backup
    
    def _restore_weights(self, checkpoint: dict):
        """Restore adapter weights from checkpoint."""
        self.adapter_trainer.model.restore_checkpoint(checkpoint)
    
    def _check_class_balance(
        self,
        class_indices: list,
        imbalance_threshold: float = 0.7,
    ) -> tuple:
        """
        Check if training data is balanced.
        
        Args:
            class_indices: List of target class indices
            imbalance_threshold: If any class > this ratio, consider imbalanced
        
        Returns:
            (is_balanced, max_ratio, dominant_class_idx)
        """
        if not class_indices:
            return True, 0.0, -1
        
        counter = {}
        for idx in class_indices:
            counter[idx] = counter.get(idx, 0) + 1
        
        total = len(class_indices)
        max_count = max(counter.values())
        max_ratio = max_count / total
        dominant_idx = [k for k, v in counter.items() if v == max_count][0]
        
        is_balanced = max_ratio < imbalance_threshold
        
        return is_balanced, max_ratio, dominant_idx
    
    def _validate_performance(
        self,
        original_probs: np.ndarray,
        threshold_drop: float = 0.05,
    ) -> tuple:
        """
        Validate adapter doesn't degrade performance.
        
        Args:
            original_probs: Ensemble probabilities for validation
            threshold_drop: Max allowed confidence drop
        
        Returns:
            (is_valid, conf_before, conf_after, drop)
        """
        conf_before, conf_after = self.adapter_trainer.evaluate_confidence(
            original_probs
        )
        drop = conf_before - conf_after
        is_valid = drop <= threshold_drop
        
        return is_valid, conf_before, conf_after, drop
    
    def _training_worker(
        self,
        ensemble_probs_list: list,
        class_indices_list: list,
        class_names: list,
        class_id_to_name: dict,
        epochs: int = 2,
        batch_size: int = 8,
        validation_probs: Optional[np.ndarray] = None,
    ):
        """
        Worker thread for adapter training.
        
        Args:
            ensemble_probs_list: List of ensemble probability vectors
            class_indices_list: List of target class indices
            class_names: List of class names for logging
            class_id_to_name: Mapping from class index to name
            epochs: Training epochs
            batch_size: Batch size
            validation_probs: Optional probs for performance validation
        """
        try:
            self.is_training = True
            print("[Adapter] Training thread started")
            
            # Check class balance
            is_balanced, max_ratio, dominant_idx = self._check_class_balance(
                class_indices_list, imbalance_threshold=0.7
            )
            
            if not is_balanced:
                dominant_name = class_id_to_name.get(dominant_idx, f"class_{dominant_idx}")
                print(
                    f"[Adapter] ⚠ Class imbalance detected: "
                    f"{dominant_name} dominates ({max_ratio:.0%})"
                )
                print(f"[Adapter] Skipping training to prevent bias")
                self.is_training = False
                return
            
            # Log class distribution
            print("[Adapter] Training data distribution:")
            counter = {}
            for idx in class_indices_list:
                counter[idx] = counter.get(idx, 0) + 1
            for idx, count in sorted(counter.items()):
                name = class_id_to_name.get(idx, f"class_{idx}")
                print(f"  {name:15s}: {count:3d} samples")
            
            # Backup weights before training
            self._backup_weights()
            print("[Adapter] Backup created")
            
            # Train
            result = self.adapter_trainer.train(
                ensemble_probs_list,
                class_indices_list,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True,
            )
            
            if not result.get('success'):
                print(f"[Adapter] Training failed: {result.get('reason')}")
                self.is_training = False
                return
            
            # Validate performance
            if validation_probs is not None:
                is_valid, conf_before, conf_after, drop = self._validate_performance(
                    validation_probs
                )
                
                print(
                    f"[Adapter] Performance check: "
                    f"Before={conf_before:.2f}, After={conf_after:.2f}, "
                    f"Drop={drop:.4f}"
                )
                
                if not is_valid:
                    print("[Adapter] ⚠ Performance drop detected, reverting...")
                    self._restore_weights(self.last_backup)
                    print("[Adapter] ✗ Adapter reverted")
                    self.is_training = False
                    return
            
            # Save adapter weights
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.adapter_weights_dir,
                f"adapter_{timestamp}.pt"
            )
            self.adapter_trainer.save_model(save_path)
            
            # Log statistics
            log_entry = {
                'timestamp': timestamp,
                'num_samples': len(ensemble_probs_list),
                'epochs': epochs,
                'status': 'success',
                'validation_passed': is_valid if validation_probs is not None else None,
            }
            self.performance_log.append(log_entry)
            
            print("[Adapter] ✓ Adapter updated successfully")
            
        except Exception as e:
            print(f"[Adapter] ✗ Training error: {e}")
            if self.last_backup:
                self._restore_weights(self.last_backup)
                print("[Adapter] Reverted to backup")
        
        finally:
            self.is_training = False
    
    def trigger_training(
        self,
        pseudo_buffer,
        classes: list,
        class_id_to_name: dict,
        epochs: int = 2,
        batch_size: int = 8,
        min_samples_per_class: int = 3,
    ) -> bool:
        """
        Trigger adapter training if conditions are met.
        
        Args:
            pseudo_buffer: PseudoLabelBuffer instance
            classes: List of all class names
            class_id_to_name: Mapping from class index to name
            epochs: Training epochs
            batch_size: Batch size
            min_samples_per_class: Minimum samples per class to train
        
        Returns:
            True if training was triggered, False otherwise
        """
        if not self.enable_adaptation:
            return False
        
        if self.is_training:
            return False  # Already training
        
        total_samples = pseudo_buffer.get_total_samples()
        
        # Check if we have enough samples
        if total_samples < 20:  # MIN_BUFFER
            return False
        
        # Check class distribution
        counts = pseudo_buffer.get_class_counts()
        classes_with_samples = sum(1 for c in counts.values() if c >= min_samples_per_class)
        
        if classes_with_samples < 2:
            return False  # Not enough classes represented
        
        # Check class balance
        is_balanced = not pseudo_buffer.check_class_imbalance(imbalance_ratio=0.7)
        if not is_balanced:
            print("[Adapter] Skipping training: class imbalance detected")
            return False
        
        # Prepare training data
        ensemble_probs_list = []
        class_indices_list = []
        
        for class_name, sequences in pseudo_buffer.buffer.items():
            # Get class index
            try:
                class_idx = classes.index(class_name)
            except ValueError:
                print(f"[Adapter] Warning: class '{class_name}' not in class list")
                continue
            
            for seq in sequences:
                # Note: We need the ensemble probs, not the sequence
                # In the webcam loop, we'll pass this directly
                ensemble_probs_list.append(seq)  # Placeholder - will be replaced
                class_indices_list.append(class_idx)
        
        if not ensemble_probs_list:
            return False
        
        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(
                ensemble_probs_list,
                class_indices_list,
                classes,
                class_id_to_name,
                epochs,
                batch_size,
                None,  # validation_probs
            ),
            daemon=True,
        )
        self.training_thread.start()
        
        return True
    
    def trigger_training_with_probs(
        self,
        ensemble_probs_list: list,
        class_indices_list: list,
        classes: list,
        class_id_to_name: dict,
        validation_probs: Optional[np.ndarray] = None,
        epochs: int = 2,
        batch_size: int = 8,
    ) -> bool:
        """
        Trigger training with ensemble probabilities (preferred).
        
        Args:
            ensemble_probs_list: List of ensemble probability vectors
            class_indices_list: List of target class indices
            classes: List of all class names
            class_id_to_name: Mapping from class index to name
            validation_probs: Optional probs for performance validation
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            True if training was triggered, False otherwise
        """
        if not self.enable_adaptation:
            return False
        
        if self.is_training:
            return False
        
        if len(ensemble_probs_list) < 20:  # MIN_BUFFER
            return False
        
        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(
                ensemble_probs_list,
                class_indices_list,
                classes,
                class_id_to_name,
                epochs,
                batch_size,
                validation_probs,
            ),
            daemon=True,
        )
        self.training_thread.start()
        
        return True
    
    def wait_for_training(self, timeout: float = 60.0) -> bool:
        """
        Wait for current training to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if training completed, False if timeout
        """
        if self.training_thread:
            self.training_thread.join(timeout=timeout)
            return not self.training_thread.is_alive()
        return True
    
    def shutdown(self):
        """Shutdown manager and wait for training to complete."""
        self.stop_training = True
        self.wait_for_training(timeout=5.0)
