"""Utility functions."""

import torch
import numpy as np
from pathlib import Path
import random
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_name="cuda"):
    """Get device."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, filename=None):
    """
    Save checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Metrics dictionary
        checkpoint_dir: Directory to save checkpoint
        filename: Filename for checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
    
    filepath = checkpoint_dir / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    return filepath


def load_checkpoint(filepath, model, optimizer=None, device="cuda"):
    """
    Load checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer (optional)
        device: Device to load to
    
    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"Checkpoint loaded from epoch {epoch}")
    
    return epoch, metrics


def save_model(model, filepath):
    """Save model weights only."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")


def load_model(model, filepath, device="cuda"):
    """Load model weights."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Model loaded from {filepath}")
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model, optimizer_name="adam", lr=1e-3, weight_decay=1e-5, **kwargs):
    """
    Get optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
    
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name="cosine", num_epochs=100, **kwargs):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler ('cosine', 'exponential', 'step')
        num_epochs: Number of epochs
    
    Returns:
        Scheduler instance
    """
    if scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, **kwargs)
    elif scheduler_name.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, **kwargs)
    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class TensorBoardLogger:
    """TensorBoard logger wrapper."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
    
    def log_scalar(self, tag, value, step):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, image, step):
        """Log image."""
        self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close writer."""
        self.writer.close()


def clip_gradients(model, max_norm=1.0):
    """Clip gradients by norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def exponential_moving_average(model, ema_model, decay=0.99):
    """
    Update exponential moving average model.
    
    Args:
        model: Current model
        ema_model: EMA model to update
        decay: Decay rate
    """
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
