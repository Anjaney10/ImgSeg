"""Configuration for U-Net cell segmentation model."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Training configuration."""
    
    # Model
    in_channels: int = 3
    out_channels: int = 1
    init_features: int = 64
    depth: int = 4
    
    # Training
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Data
    image_size: int = 256
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd
    scheduler: str = "cosine"  # cosine, exponential, step
    warmup_epochs: int = 5
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    output_dir: Path = Path("outputs")
    save_frequency: int = 10  # Save every N epochs
    
    # Paths
    data_dir: Path = Path("data")
    train_images_dir: Path = Path("data/train/images")
    train_masks_dir: Path = Path("data/train/masks")
    val_images_dir: Path = Path("data/val/images")
    val_masks_dir: Path = Path("data/val/masks")
    
    # Inference
    confidence_threshold: float = 0.5
    post_process: bool = True
    
    # Misc
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Default configuration
DEFAULT_CONFIG = Config()
