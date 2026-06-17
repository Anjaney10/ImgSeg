"""Training script for U-Net cell segmentation model."""

import os
import sys
import argparse
from pathlib import Path
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.unet import create_unet
from src.dataset import create_data_loaders
from src.losses import get_loss_function
from src.metrics import SegmentationMetrics, calculate_dice, calculate_iou
from src.utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    save_model,
    count_parameters,
    get_optimizer,
    get_scheduler,
    clip_gradients,
    TensorBoardLogger,
)


class Trainer:
    """Trainer class for U-Net model."""
    
    def __init__(self, config):
        """Initialize trainer."""
        self.config = config
        self.device = get_device(config.device)
        
        # Set seed
        set_seed(config.seed)
        
        # Create model
        self.model = create_unet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_features=config.init_features,
            depth=config.depth,
        ).to(self.device)
        
        print(f"Model created with {count_parameters(self.model):,} parameters")
        
        # Loss function
        self.criterion = get_loss_function("bce_dice", bce_weight=0.5, dice_weight=0.5)
        
        # Optimizer
        self.optimizer = get_optimizer(
            self.model,
            optimizer_name=config.optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_name=config.scheduler,
            num_epochs=config.num_epochs,
        )
        
        # Logger
        self.logger = TensorBoardLogger(log_dir=config.log_dir)
        
        # Metrics
        self.train_metrics = SegmentationMetrics(threshold=0.5)
        self.val_metrics = SegmentationMetrics(threshold=0.5)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                clip_gradients(self.model, self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs.detach(), masks.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Get metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self.train_metrics.get_metrics()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Update metrics
            total_loss += loss.item()
            self.val_metrics.update(outputs, masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # Get metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.val_metrics.get_metrics()
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader):
        """Train model."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Metrics: Dice={train_metrics['dice']:.4f}, IoU={train_metrics['iou']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}")
            
            # TensorBoard logging
            self.logger.log_scalar('Loss/train', train_loss, epoch)
            self.logger.log_scalar('Loss/val', val_loss, epoch)
            self.logger.log_scalars('Metrics/train', train_metrics, epoch)
            self.logger.log_scalars('Metrics/val', val_metrics, epoch)
            
            # Save checkpoint
            checkpoint_data = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'epoch': epoch,
            }
            
            if (epoch + 1) % self.config.save_frequency == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    checkpoint_data,
                    self.config.checkpoint_dir,
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(self.model, str(self.config.checkpoint_dir / "best_model.pth"))
            
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Dice: {self.best_val_dice:.4f}")
        print(f"{'='*60}\n")
        
        # Close logger
        self.logger.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train U-Net for cell segmentation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-dir', type=str, default='data/train/images', help='Training images directory')
    parser.add_argument('--val-dir', type=str, default='data/val/images', help='Validation images directory')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = args.device
    config.seed = args.seed
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(
            config.train_images_dir,
            config.train_masks_dir,
            config.val_images_dir,
            config.val_masks_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            augmentation_prob=config.augmentation_prob,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data is organized as:")
        print("  data/train/images/  - training images")
        print("  data/train/masks/   - training masks")
        print("  data/val/images/    - validation images")
        print("  data/val/masks/     - validation masks")
        return
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
