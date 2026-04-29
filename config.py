"""
Production-grade Configuration Module for ISL (Indian Sign Language) Word Recognition Pipeline.

## Overview
Provides a structured, validated, and self-documenting configuration system organized into
logical sections: Paths, Feature Dimensions, Model Architecture, Training, Inference, and Hardware.

## Design Principles
- **Modularity**: Feature dimension computation is explicit and decoupled from toggles
- **Validation**: Configuration consistency is verified at startup to prevent silent errors
- **Clarity**: All feature calculations are visible and explained
- **Robustness**: Resolution-independent motion thresholds and safe CPU thread management
- **Research-Grade**: CONFIG_VERSION tracks changes, DEBUG mode aids development

## Usage
    from config import get_config
    cfg = get_config()
    print(cfg.summary())  # Print dimension breakdown
    cfg.validate()        # Validate consistency (auto-called at module load)
"""

import os
import sys
import hashlib
from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

# ========================================================================================
# ─────── CONFIG VERSION & DEBUG MODE ──────────────────────────────────────
# ========================================================================================

CONFIG_VERSION = "2.0.0"  # Bump when making breaking config changes
DEBUG_MODE = False        # Set to True for verbose config diagnostics


# ========================================================================================
# ─────── PATHS CONFIGURATION ──────────────────────────────────────────────
# ========================================================================================

@dataclass
class PathsConfig:
    """File and directory paths for the ISL pipeline."""

    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    dataset_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Dataset"))
    processed_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "processed"))
    model_save_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model.pth"))
    ensemble_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ensemble"))
    hand_landmarker_model: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"))
    face_landmarker_model: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"))

    video_extensions: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")
    num_folds: int = 5  # K-fold cross-validation


# ========================================================================================
# ─────── LANDMARK & FEATURE DIMENSION CONFIGURATION ──────────────────────
# ========================================================================================

@dataclass
class LandmarkConfig:
    """Landmark detection and raw feature dimensions."""

    num_landmarks: int = 21       # MediaPipe hand landmarks per hand
    num_coords: int = 3           # (x, y, z) coordinates per landmark
    num_hands: int = 2            # Capture both hands (right=0, left=1)

    # Computed properties (do not modify directly)
    @property
    def landmark_dim_per_hand(self) -> int:
        """Dimension per hand: landmarks × coordinates."""
        return self.num_landmarks * self.num_coords

    @property
    def raw_frame_features_dim(self) -> int:
        """Raw hand coordinates per frame: both hands concatenated."""
        return self.landmark_dim_per_hand * self.num_hands


@dataclass
class SpatialFeaturesConfig:
    """Spatial and relative hand position features."""

    use_face_relative: bool = True
    """Compute hand coordinates relative to face anchor points (nose center)."""

    use_spatial_distance: bool = False
    """Compute pairwise distances between landmarks (distance matrix per hand)."""

    # Computed properties
    @property
    def spatial_distance_dim_per_hand(self) -> int:
        """Dimension if spatial distance enabled: landmarks × 4 (upper tri of distance matrix)."""
        return 21 * 4 if self.use_spatial_distance else 0

    @property
    def relative_frame_features_dim(self) -> int:
        """Relative hand features per frame (face-anchored coordinates + distances)."""
        landmarks = LandmarkConfig()
        per_hand = landmarks.landmark_dim_per_hand + self.spatial_distance_dim_per_hand
        return per_hand * landmarks.num_hands if self.use_face_relative else 0

    @property
    def proximity_dim(self) -> int:
        """Hand-to-face proximity scalar (distance metric)."""
        return 1 if self.use_face_relative else 0


@dataclass
class FrameFeaturesConfig:
    """Complete per-frame feature vector structure."""

    landmark_cfg: LandmarkConfig = field(default_factory=LandmarkConfig)
    spatial_cfg: SpatialFeaturesConfig = field(default_factory=SpatialFeaturesConfig)

    use_velocity: bool = True
    """Include frame-to-frame velocity (delta) features."""

    @property
    def frame_features_dim(self) -> int:
        """Total features per frame (without velocity)."""
        raw = self.landmark_cfg.raw_frame_features_dim
        relative = self.spatial_cfg.relative_frame_features_dim
        proximity = self.spatial_cfg.proximity_dim
        return raw + relative + proximity

    @property
    def input_sequence_dim(self) -> int:
        """Sequence input dimension: frame features × 2 if velocity enabled."""
        base_dim = self.frame_features_dim
        return base_dim * 2 if self.use_velocity else base_dim

    @property
    def proximity_index(self) -> int:
        """Index of proximity scalar in feature vector (-1 if not present)."""
        if self.spatial_cfg.proximity_dim == 0:
            return -1
        return self.frame_features_dim - 1


# ========================================================================================
# ─────── PREPROCESSING & EXTRACTION CONFIGURATION ──────────────────────
# ========================================================================================

@dataclass
class PreprocessingConfig:
    """Video preprocessing and frame extraction settings."""

    num_frames: int = 20
    """Number of frames sampled per video. Must match webcam capture."""

    webcam_width: int = 640
    """Webcam/video frame width in pixels."""

    webcam_height: int = 480
    """Webcam/video frame height in pixels."""

    crop_to_webcam_size: bool = True
    """Center-crop all videos to webcam dimensions for consistency."""

    # Face landmark indices (MediaPipe Face model)
    face_nose_index: int = 1
    face_left_eye_index: int = 33
    face_right_eye_index: int = 263

    debug_draw_face_center: bool = True
    """Visualize face anchor point during debugging."""

    @property
    def frame_area(self) -> int:
        """Total pixel area of a frame."""
        return self.webcam_width * self.webcam_height

    @property
    def diagonal_pixels(self) -> float:
        """Frame diagonal in pixels (used for resolution-independent scaling)."""
        return (self.webcam_width**2 + self.webcam_height**2) ** 0.5


# ========================================================================================
# ─────── MODEL ARCHITECTURE CONFIGURATION ──────────────────────────────
# ========================================================================================

@dataclass
class ModelConfig:
    """Neural network architecture hyperparameters."""

    hidden_size: int = 256
    """Hidden dimension of LSTM/GRU layers."""

    num_layers: int = 3
    """Number of recurrent layers (depth of the network)."""

    bidirectional: bool = True
    """Use bidirectional recurrent network."""

    dropout: float = 0.25
    """Dropout rate for regularization (0-1)."""

    use_face_proximity_attention: bool = True
    """Apply attention weighting based on hand-to-face proximity."""

    proximity_sigma: float = 0.15
    """Standard deviation of the Gaussian proximity kernel over normalized distance."""

    learnable_proximity_sigma: bool = True
    """Allow proximity_sigma to be learned during training."""

    def validate(self) -> None:
        """Validate model configuration consistency."""
        assert 0 < self.dropout < 1.0, "Dropout must be in (0, 1)"
        assert self.hidden_size > 0, "Hidden size must be positive"
        assert self.num_layers > 0, "Number of layers must be positive"
        assert 0 < self.proximity_sigma < 1.0, "Proximity sigma must be in (0, 1)"


# ========================================================================================
# ─────── TRAINING CONFIGURATION ───────────────────────────────────────
# ========================================================================================

@dataclass
class TrainingConfig:
    """Training loop and optimization hyperparameters."""

    # Optimization
    batch_size: int = 16
    learning_rate: float = 3e-4
    """Reduced from 5e-4 for improved stability with small datasets."""

    weight_decay: float = 5e-4
    """L2 regularization strength. Increased for regularization."""

    grad_clip: float = 1.0
    """Gradient clipping max norm to prevent explosion."""

    # Epochs and early stopping
    num_epochs: int = 60
    """Increased from 40 for more stable convergence with limited data."""

    patience: int = 10
    """Early stopping patience (reduced from 20 to prevent overfitting)."""

    scheduler_patience: int = 5
    """Learning rate scheduler patience (more aggressive LR reduction)."""

    # Data splits
    val_split: float = 0.25
    """Train/validation split ratio (80/20)."""

    random_seed: int = 42

    # Label smoothing
    label_smoothing: float = 0.05
    """Smoothing factor for label distributions (improves robustness)."""

    # Class weighting
    use_class_weights: bool = True
    """Weight loss by inverse class frequency for imbalanced datasets."""

    class_weight_power: float = 1.0
    """Smoothing exponent for class weights.
    
    Values:
        - 0.5: smoother, less aggressive
        - 0.7: moderate
        - 1.0: full inverse frequency (aggressive)
    """

    # Focal loss (hard sample mining)
    use_focal_loss: bool = False
    """Use focal loss instead of standard cross-entropy."""

    focal_alpha: float = 0.25
    """Focal loss class weighting factor."""

    focal_gamma: float = 2.0
    """Focal loss hard-example focusing parameter (0=CE, 2.0=strong focus)."""

    # Mixup & CutMix augmentation
    use_mixup: bool = True
    """Apply mixup data augmentation during training."""

    use_cutmix: bool = False
    """Apply CutMix augmentation (disable if using mixup only)."""

    mixup_alpha: float = 0.3
    """Beta distribution parameter for mixup."""

    mixup_prob: float = 0.5
    """Probability of applying mixup to a batch."""

    # Learning rate scheduling
    lr_scheduler: str = "cosine"
    """Scheduler type: 'cosine', 'step', or 'exponential'."""

    lr_decay_factor: float = 0.1
    """Decay factor for step scheduler."""

    lr_min: float = 1e-5
    """Minimum learning rate for cosine annealing."""

    warmup_epochs: int = 2
    """Number of warmup epochs with linear LR increase."""

    def validate(self) -> None:
        """Validate training configuration consistency."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 < self.learning_rate < 1.0, "Learning rate should be in (0, 1)"
        assert 0 <= self.label_smoothing < 1.0, "Label smoothing must be in [0, 1)"
        assert 0 < self.val_split < 1.0, "Validation split must be in (0, 1)"
        assert self.patience > 0, "Patience must be positive"
        assert self.lr_scheduler in ("cosine", "step", "exponential"), \
            f"Unknown scheduler: {self.lr_scheduler}"


# ========================================================================================
# ─────── INFERENCE & PREDICTION CONFIGURATION ─────────────────────────
# ========================================================================================

@dataclass
class InferenceConfig:
    """Inference-time settings for prediction and confidence thresholding."""

    confidence_threshold: float = 0.40
    """Base confidence threshold for predictions (dynamically adjusted)."""

    prediction_smoothing_window: int = 5
    """Majority vote window size for temporal smoothing (smaller = faster transitions)."""

    transition_hysteresis: float = 0.12
    """Minimum confidence delta to trigger prediction switch (prevents jitter)."""

    sign_idle_timeout: int = 30
    """Frames before considering hands idle (≈ 1 second @ 30fps)."""

    similar_class_penalty: float = 0.08
    """Extra threshold penalty for easily-confused sign classes."""

    def validate(self) -> None:
        """Validate inference configuration."""
        assert 0 < self.confidence_threshold < 1.0, "Confidence threshold must be in (0, 1)"
        assert self.prediction_smoothing_window > 0, "Smoothing window must be positive"
        assert 0 < self.transition_hysteresis < 1.0, "Hysteresis must be in (0, 1)"


# ========================================================================================
# ─────── MOTION DETECTION & GATING CONFIGURATION ──────────────────────
# ========================================================================================

@dataclass
class MotionConfig:
    """Hand motion detection and gating settings (resolution-independent)."""

    enabled: bool = True
    """Enable motion-based prediction gating."""

    # Motion threshold (normalized to frame diagonal)
    motion_threshold_normalized: float = 0.015
    """Motion threshold as fraction of frame diagonal.
    
    For 640×480: diagonal ≈ 829 pixels → threshold ≈ 12.4 pixels
    For 1920×1080: diagonal ≈ 2203 pixels → threshold ≈ 33 pixels
    Auto-scales to different resolutions.
    """

    motion_smoothing: float = 0.7
    """Exponential moving average factor for motion detection (0-1)."""

    idle_confidence_threshold: float = 0.70
    """Higher confidence requirement when hands are stationary."""

    # Dynamic thresholds based on motion
    dynamic_threshold_enabled: bool = True
    """Adjust base threshold based on motion and stability."""

    motion_boost_factor: float = 0.15
    """Reduce threshold by this amount when motion is detected (encourages recognition)."""

    stability_boost_factor: float = 0.10
    """Reduce threshold as sign becomes more stable (cumulative over time)."""

    dynamic_threshold_min: float = 0.20
    """Never reduce threshold below this floor."""

    def get_motion_threshold_pixels(self, frame_width: int, frame_height: int) -> float:
        """Compute motion threshold in pixels for given frame resolution.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Motion threshold in pixels (resolution-independent)
        """
        diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
        return self.motion_threshold_normalized * diagonal

    def validate(self) -> None:
        """Validate motion configuration."""
        assert 0 < self.motion_threshold_normalized < 1.0, \
            "Normalized motion threshold must be in (0, 1)"
        assert 0 < self.motion_smoothing < 1.0, "Motion smoothing must be in (0, 1)"
        assert 0 < self.idle_confidence_threshold < 1.0, \
            "Idle threshold must be in (0, 1)"
        assert self.motion_boost_factor >= 0, "Motion boost must be non-negative"
        assert self.stability_boost_factor >= 0, "Stability boost must be non-negative"
        assert 0 < self.dynamic_threshold_min < 1.0, \
            "Dynamic threshold min must be in (0, 1)"


# ========================================================================================
# ─────── HARDWARE & DEVICE CONFIGURATION ──────────────────────────────
# ========================================================================================

@dataclass
class HardwareConfig:
    """Device and CPU optimization settings."""

    device_type: str = "cpu"
    """Device: 'cpu' or 'cuda'."""

    num_threads: int = field(default_factory=lambda: max(1, (os.cpu_count() or 1) - 1))
    """Number of CPU threads (auto-detected from os.cpu_count(), with safe fallback)."""

    @property
    def torch_device(self) -> torch.device:
        """Return PyTorch device object."""
        return torch.device(self.device_type)

    def apply_torch_settings(self) -> None:
        """Apply thread and device settings to PyTorch."""
        torch.set_num_threads(self.num_threads)
        if self.device_type == "cpu" and DEBUG_MODE:
            print(f"[Config] PyTorch: {self.num_threads} threads on CPU")

    def validate(self) -> None:
        """Validate hardware configuration."""
        assert self.num_threads > 0, "Number of threads must be positive"
        assert self.device_type in ("cpu", "cuda"), f"Unknown device: {self.device_type}"


# ========================================================================================
# ─────── WEBCAM CAPTURE CONFIGURATION ─────────────────────────────────
# ========================================================================================

@dataclass
class WebcamConfig:
    """Real-time webcam capture settings."""

    record_frames: int = 90
    """Raw frames to capture before sub-sampling to NUM_FRAMES."""

    countdown: int = 3
    """Countdown seconds before recording starts."""


# ========================================================================================
# ─────── UNIFIED CONFIGURATION CLASS ──────────────────────────────────
# ========================================================================================

@dataclass
class Config:
    """Master configuration class integrating all subsystems.
    
    Provides a single point of access for all pipeline parameters with
    automatic validation and computed dimension properties.
    """

    version: str = CONFIG_VERSION
    debug: bool = DEBUG_MODE

    # Subsystem configs
    paths: PathsConfig = field(default_factory=PathsConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    spatial: SpatialFeaturesConfig = field(default_factory=SpatialFeaturesConfig)
    frame_features: FrameFeaturesConfig = field(default_factory=FrameFeaturesConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    webcam: WebcamConfig = field(default_factory=WebcamConfig)

    def validate(self) -> None:
        """Validate all configuration subsystems and cross-module consistency.
        
        Raises:
            AssertionError: If configuration is invalid or inconsistent.
        """
        # Individual subsystem validations
        self.model.validate()
        self.training.validate()
        self.inference.validate()
        self.motion.validate()
        self.hardware.validate()

        # Cross-module consistency checks
        self._validate_feature_dimensions()

    def _validate_feature_dimensions(self) -> None:
        """Verify feature dimension consistency across all toggles."""
        # Ensure feature flags match final dimensions
        if self.spatial.use_face_relative:
            assert self.frame_features.spatial_cfg.relative_frame_features_dim > 0, \
                "USE_FACE_RELATIVE=True but relative features not computed"
        else:
            assert self.frame_features.spatial_cfg.relative_frame_features_dim == 0, \
                "USE_FACE_RELATIVE=False but relative features present"

        if self.spatial.use_spatial_distance:
            assert self.frame_features.spatial_cfg.spatial_distance_dim_per_hand > 0, \
                "USE_SPATIAL_DISTANCE=True but distance features not computed"
        else:
            assert self.frame_features.spatial_cfg.spatial_distance_dim_per_hand == 0, \
                "USE_SPATIAL_DISTANCE=False but distance features present"

        if self.frame_features.use_velocity:
            assert self.frame_features.input_sequence_dim == self.frame_features.frame_features_dim * 2, \
                "USE_VELOCITY=True but input dimension not doubled"
        else:
            assert self.frame_features.input_sequence_dim == self.frame_features.frame_features_dim, \
                "USE_VELOCITY=False but input dimension not matching frame features"

        # Sanity checks on computed dimensions
        assert self.frame_features.frame_features_dim > 0, \
            "Frame features dimension is zero (invalid configuration)"
        assert self.frame_features.input_sequence_dim > 0, \
            "Input sequence dimension is zero (invalid configuration)"

        if self.debug:
            print("[Config] ✓ Feature dimension consistency validated")

    def get_motion_threshold_pixels(self) -> float:
        """Convenience method to get motion threshold in pixels."""
        return self.motion.get_motion_threshold_pixels(
            self.preprocessing.webcam_width,
            self.preprocessing.webcam_height
        )

    def compute_config_hash(self) -> str:
        """Compute SHA256 hash of configuration for reproducibility tracking.
        
        Returns:
            Hex string of config hash (useful for experiment tracking).
        """
        import json
        # Create a serializable dict of key parameters
        config_dict = {
            "version": self.version,
            "landmarks": {
                "num_landmarks": self.landmarks.num_landmarks,
                "num_coords": self.landmarks.num_coords,
                "num_hands": self.landmarks.num_hands,
            },
            "spatial": {
                "use_face_relative": self.spatial.use_face_relative,
                "use_spatial_distance": self.spatial.use_spatial_distance,
            },
            "frame_features": {
                "use_velocity": self.frame_features.use_velocity,
            },
            "preprocessing": {
                "num_frames": self.preprocessing.num_frames,
                "webcam_width": self.preprocessing.webcam_width,
                "webcam_height": self.preprocessing.webcam_height,
            },
            "model": {
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
            },
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    def summary(self) -> str:
        """Generate a human-readable summary of key configuration parameters.
        
        Returns:
            Formatted string with dimension breakdown and key settings.
        """
        motion_threshold_px = self.get_motion_threshold_pixels()
        config_hash = self.compute_config_hash()

        summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ISL PIPELINE CONFIGURATION SUMMARY                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

[Version & Reproducibility]
  Config Version: {self.version}
  Config Hash: {config_hash}
  Debug Mode: {self.debug}

[Feature Dimensions per Frame]
  Landmark features (raw): {self.landmarks.raw_frame_features_dim}
    └─ ({self.landmarks.num_landmarks} landmarks × {self.landmarks.num_coords} coords × {self.landmarks.num_hands} hands)
  Spatial relative features: {self.spatial.relative_frame_features_dim}
    └─ Face-relative: {self.spatial.use_face_relative} | Distance matrix: {self.spatial.use_spatial_distance}
  Proximity features: {self.spatial.proximity_dim}
  ─────────────────────────────────
  Total per frame: {self.frame_features.frame_features_dim}

[Sequence Input]
  Frames: {self.preprocessing.num_frames}
  Use velocity: {self.frame_features.use_velocity}
  Input dimension: {self.frame_features.input_sequence_dim}
    ➜ Sequence shape: (batch, {self.preprocessing.num_frames}, {self.frame_features.input_sequence_dim})

[Motion Detection (Resolution-Independent)]
  Enabled: {self.motion.enabled}
  Frame resolution: {self.preprocessing.webcam_width}×{self.preprocessing.webcam_height}
  Motion threshold (normalized): {self.motion.motion_threshold_normalized:.4f} × diagonal
  Motion threshold (pixels): {motion_threshold_px:.2f}
  Idle confidence threshold: {self.motion.idle_confidence_threshold}

[Model Architecture]
  Recurrent type: LSTM/GRU
  Hidden size: {self.model.hidden_size}
  Layers: {self.model.num_layers} (bidirectional: {self.model.bidirectional})
  Dropout: {self.model.dropout}
  Proximity attention: {self.model.use_face_proximity_attention}

[Training]
  Batch size: {self.training.batch_size}
  Learning rate: {self.training.learning_rate}
  Epochs: {self.training.num_epochs}
  Early stopping patience: {self.training.patience}
  Label smoothing: {self.training.label_smoothing}
  Class weighting: {self.training.use_class_weights} (power={self.training.class_weight_power})

[Inference]
  Confidence threshold: {self.inference.confidence_threshold}
  Smoothing window: {self.inference.prediction_smoothing_window}
  Transition hysteresis: {self.inference.transition_hysteresis}

[Hardware]
  Device: {self.hardware.device_type.upper()}
  CPU threads: {self.hardware.num_threads}

╔═══════════════════════════════════════════════════════════════════════════════╗
"""
        return summary

    def __post_init__(self):
        """Initialize subsystem cross-references after dataclass initialization."""
        # Ensure frame features uses the same landmark config
        self.frame_features.landmark_cfg = self.landmarks
        self.frame_features.spatial_cfg = self.spatial


# ========================================================================================
# ─────── MODULE-LEVEL SINGLETON & INITIALIZATION ──────────────────────
# ========================================================================================

_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance (singleton pattern).
    
    Returns:
        Config: Validated configuration object.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _config_instance.validate()
        _config_instance.hardware.apply_torch_settings()
        if DEBUG_MODE or True:  # Always print on module init
            print(_config_instance.summary())
    return _config_instance


def reset_config() -> None:
    """Reset global configuration instance (useful for testing)."""
    global _config_instance
    _config_instance = None


# ========================================================================================
# ─────── BACKWARD COMPATIBILITY EXPORTS ──────────────────────────────
# ========================================================================================

# Initialize config on module import
_default_config = get_config()

# Export module-level constants for backward compatibility with existing code
BASE_DIR = _default_config.paths.base_dir
DATASET_DIR = _default_config.paths.dataset_dir
PROCESSED_DIR = _default_config.paths.processed_dir
MODEL_SAVE_PATH = _default_config.paths.model_save_path
ENSEMBLE_DIR = _default_config.paths.ensemble_dir
NUM_FOLDS = _default_config.paths.num_folds

NUM_FRAMES = _default_config.preprocessing.num_frames
WEBCAM_WIDTH = _default_config.preprocessing.webcam_width
WEBCAM_HEIGHT = _default_config.preprocessing.webcam_height
CROP_TO_WEBCAM_SIZE = _default_config.preprocessing.crop_to_webcam_size
NUM_LANDMARKS = _default_config.landmarks.num_landmarks
NUM_COORDS = _default_config.landmarks.num_coords
NUM_HANDS = _default_config.landmarks.num_hands
LANDMARK_DIM = _default_config.landmarks.landmark_dim_per_hand
RAW_FRAME_FEAT_DIM = _default_config.landmarks.raw_frame_features_dim
USE_FACE_RELATIVE = _default_config.spatial.use_face_relative
USE_SPATIAL_DISTANCE = _default_config.spatial.use_spatial_distance
SPATIAL_DISTANCE_DIM = _default_config.spatial.spatial_distance_dim_per_hand
REL_FRAME_FEAT_DIM = _default_config.spatial.relative_frame_features_dim
PROXIMITY_FEAT_DIM = _default_config.spatial.proximity_dim
FRAME_FEAT_DIM = _default_config.frame_features.frame_features_dim
PROXIMITY_INDEX = _default_config.frame_features.proximity_index
USE_VELOCITY = _default_config.frame_features.use_velocity
INPUT_SIZE = _default_config.frame_features.input_sequence_dim
VIDEO_EXTENSIONS = _default_config.paths.video_extensions
HAND_LANDMARKER_MODEL = _default_config.paths.hand_landmarker_model
FACE_LANDMARKER_MODEL = _default_config.paths.face_landmarker_model
FACE_NOSE_INDEX = _default_config.preprocessing.face_nose_index
FACE_LEFT_EYE_INDEX = _default_config.preprocessing.face_left_eye_index
FACE_RIGHT_EYE_INDEX = _default_config.preprocessing.face_right_eye_index
DEBUG_DRAW_FACE_CENTER = _default_config.preprocessing.debug_draw_face_center

HIDDEN_SIZE = _default_config.model.hidden_size
NUM_LAYERS = _default_config.model.num_layers
BIDIRECTIONAL = _default_config.model.bidirectional
DROPOUT = _default_config.model.dropout
USE_FACE_PROXIMITY_ATTENTION = _default_config.model.use_face_proximity_attention
PROXIMITY_SIGMA = _default_config.model.proximity_sigma
LEARNABLE_PROXIMITY_SIGMA = _default_config.model.learnable_proximity_sigma

BATCH_SIZE = _default_config.training.batch_size
NUM_EPOCHS = _default_config.training.num_epochs
LEARNING_RATE = _default_config.training.learning_rate
WEIGHT_DECAY = _default_config.training.weight_decay
LABEL_SMOOTHING = _default_config.training.label_smoothing
PATIENCE = _default_config.training.patience
SCHEDULER_PATIENCE = _default_config.training.scheduler_patience
GRAD_CLIP = _default_config.training.grad_clip
VAL_SPLIT = _default_config.training.val_split
RANDOM_SEED = _default_config.training.random_seed
USE_CLASS_WEIGHTS = _default_config.training.use_class_weights
CLASS_WEIGHT_POWER = _default_config.training.class_weight_power
USE_FOCAL_LOSS = _default_config.training.use_focal_loss
FOCAL_ALPHA = _default_config.training.focal_alpha
FOCAL_GAMMA = _default_config.training.focal_gamma
USE_MIXUP = _default_config.training.use_mixup
USE_CUTMIX = _default_config.training.use_cutmix
MIXUP_ALPHA = _default_config.training.mixup_alpha
MIXUP_PROB = _default_config.training.mixup_prob
LR_SCHEDULER = _default_config.training.lr_scheduler
LR_DECAY_FACTOR = _default_config.training.lr_decay_factor
LR_MIN = _default_config.training.lr_min
WARMUP_EPOCHS = _default_config.training.warmup_epochs

DEVICE = _default_config.hardware.torch_device
NUM_THREADS = _default_config.hardware.num_threads

WEBCAM_RECORD_FRAMES = _default_config.webcam.record_frames
WEBCAM_COUNTDOWN = _default_config.webcam.countdown

CONFIDENCE_THRESHOLD = _default_config.inference.confidence_threshold
PREDICTION_SMOOTHING_WINDOW = _default_config.inference.prediction_smoothing_window

MOTION_GATING_ENABLED = _default_config.motion.enabled
MOTION_THRESHOLD = _default_config.get_motion_threshold_pixels()
MOTION_SMOOTHING = _default_config.motion.motion_smoothing
IDLE_CONFIDENCE_THRESHOLD = _default_config.motion.idle_confidence_threshold

DYNAMIC_THRESHOLD_ENABLED = _default_config.motion.dynamic_threshold_enabled
MOTION_BOOST_FACTOR = _default_config.motion.motion_boost_factor
STABILITY_BOOST_FACTOR = _default_config.motion.stability_boost_factor
DYNAMIC_THRESHOLD_MIN = _default_config.motion.dynamic_threshold_min

TRANSITION_HYSTERESIS = _default_config.inference.transition_hysteresis
SIGN_IDLE_TIMEOUT = _default_config.inference.sign_idle_timeout
SIMILAR_CLASS_PENALTY = _default_config.inference.similar_class_penalty
