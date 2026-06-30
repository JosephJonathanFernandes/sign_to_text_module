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
import os
from typing import Optional
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

    def _balance_training_data(
        self,
        ensemble_probs_list: list,
        class_indices_list: list,
        min_samples_per_class: int = 5,
        max_samples_per_class: int = 100,
    ) -> tuple:
        """
        Build a balanced training set by capping each eligible class to
        max_samples_per_class, instead of strictly downsampling everything
        to the smallest class size.

        Returns:
            (balanced_probs, balanced_targets, counts)
        """
        if len(ensemble_probs_list) != len(class_indices_list):
            raise ValueError("Mismatched probs/targets length")

        by_class = {}
        for probs, class_idx in zip(ensemble_probs_list, class_indices_list):
            by_class.setdefault(class_idx, []).append(probs)

        counts = {class_idx: len(samples) for class_idx, samples in by_class.items()}
        eligible = {
            class_idx: samples
            for class_idx, samples in by_class.items()
            if len(samples) >= min_samples_per_class
        }

        if len(eligible) < 3:
            return [], [], counts

        balanced_probs = []
        balanced_targets = []
        for class_idx in sorted(eligible):
            samples = eligible[class_idx]
            limit = min(len(samples), max_samples_per_class)
            
            if len(samples) > limit:
                chosen = np.random.choice(len(samples), limit, replace=False)
                selected = [samples[i] for i in chosen]
            else:
                selected = list(samples)

            balanced_probs.extend(selected)
            balanced_targets.extend([class_idx] * len(selected))

        return balanced_probs, balanced_targets, counts
    
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
        min_samples_per_class: int = 5,
        use_class_weights: bool = True,
        class_weight_power: float = 0.5,
        class_weight_clip_min: float = 0.5,
        class_weight_clip_max: float = 3.0,
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

            balanced_probs, balanced_targets, raw_counts = self._balance_training_data(
                ensemble_probs_list,
                class_indices_list,
                min_samples_per_class=min_samples_per_class,
            )

            if not balanced_probs:
                print(
                    "[Adapter] Skipping training: not enough balanced samples "
                    f"(need >= {min_samples_per_class} per class and at least 3 classes)"
                )
                self.is_training = False
                return

            if len(balanced_probs) != len(balanced_targets):
                print("[Adapter] Skipping training: balanced dataset construction failed")
                self.is_training = False
                return

            # Compute mild inverse-frequency class weights from the raw saved distribution.
            # These are normalized and clipped so they help with bias without causing instability.
            class_weights = {}
            if use_class_weights:
                eligible_counts = {
                    idx: count
                    for idx, count in raw_counts.items()
                    if count >= min_samples_per_class
                }
                if eligible_counts:
                    exponent = max(0.0, float(class_weight_power))
                    raw_weights = {
                        idx: (1.0 / (float(count) ** exponent)) if exponent > 0.0 else 1.0
                        for idx, count in eligible_counts.items()
                    }
                    mean_weight = float(np.mean(list(raw_weights.values())))
                    lower = float(class_weight_clip_min)
                    upper = float(class_weight_clip_max)
                    for idx, raw_weight in raw_weights.items():
                        normalized = raw_weight / mean_weight if mean_weight > 0 else 1.0
                        class_weights[idx] = float(np.clip(normalized, lower, upper))
            
            # Check class balance
            is_balanced, max_ratio, dominant_idx = self._check_class_balance(
                balanced_targets, imbalance_threshold=0.7
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
            print("[Adapter] Raw data distribution:")
            for idx, count in sorted(raw_counts.items()):
                name = class_id_to_name.get(idx, f"class_{idx}")
                print(f"  {name:15s}: {count:3d} samples")

            print("[Adapter] Balanced training distribution:")
            balanced_counter = {}
            for idx in balanced_targets:
                balanced_counter[idx] = balanced_counter.get(idx, 0) + 1
            for idx, count in sorted(balanced_counter.items()):
                name = class_id_to_name.get(idx, f"class_{idx}")
                print(f"  {name:15s}: {count:3d} samples")
            
            # Backup weights before training
            self._backup_weights()
            print("[Adapter] Backup created")
            
            # Train
            result = self.adapter_trainer.train(
                balanced_probs,
                balanced_targets,
                class_weights=class_weights,
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
                'balanced_samples': len(balanced_probs),
                'class_weights': class_weights,
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
        min_classes: int = 3,
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
            print("[Adapter] Skipping training: adaptation is disabled")
            return False
        
        if self.is_training:
            print("[Adapter] Skipping training: adapter is already training")
            return False  # Already training
        
        total_samples = pseudo_buffer.get_total_samples()
        
        # Check if we have enough samples
        if total_samples < 20:  # MIN_BUFFER
            print(f"[Adapter] Skipping training: only {total_samples} samples available")
            return False
        
        # Check class distribution
        counts = pseudo_buffer.get_class_counts()
        classes_with_samples = sum(1 for c in counts.values() if c >= min_samples_per_class)
        
        if classes_with_samples < min_classes:
            print(
                f"[Adapter] Skipping training: need at least {min_classes} classes with "
                f">= {min_samples_per_class} samples each"
            )
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
        min_classes: int = 3,
        min_samples_per_class: int = 5,
        use_class_weights: bool = True,
        class_weight_power: float = 0.5,
        class_weight_clip_min: float = 0.5,
        class_weight_clip_max: float = 3.0,
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
            print("[Adapter] Skipping training: adaptation is disabled")
            return False
        
        if self.is_training:
            print("[Adapter] Skipping training: adapter is already training")
            return False
        
        if len(ensemble_probs_list) < 20:  # MIN_BUFFER
            print(
                f"[Adapter] Skipping training: only {len(ensemble_probs_list)} samples available"
            )
            return False

        counts = {}
        for class_idx in class_indices_list:
            counts[class_idx] = counts.get(class_idx, 0) + 1

        eligible_classes = [
            class_idx for class_idx, count in counts.items()
            if count >= min_samples_per_class
        ]

        if len(eligible_classes) < min_classes:
            print(
                f"[Adapter] Skipping training: need at least {min_classes} classes with "
                f">= {min_samples_per_class} samples each"
            )
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
                min_samples_per_class,
                use_class_weights,
                class_weight_power,
                class_weight_clip_min,
                class_weight_clip_max,
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
