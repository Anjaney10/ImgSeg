"""Dataset utilities for cell segmentation."""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CellSegmentationDataset(Dataset):
    """Cell segmentation dataset."""
    
    def __init__(
        self,
        images_dir,
        masks_dir,
        image_size=256,
        augmentation=None,
        normalize=True,
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Path to directory containing images
            masks_dir: Path to directory containing masks
            image_size: Size to resize images to
            augmentation: Albumentations augmentation pipeline
            normalize: Whether to normalize images
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        self.normalize = normalize
        
        # Get list of image files
        self.image_files = sorted([f for f in self.images_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        
        # Load mask
        mask_path = self.masks_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.float32)
        
        # Ensure mask is binary
        mask = (mask > 127.5).astype(np.float32)
        
        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default: resize and normalize
            image = Image.fromarray((image).astype(np.uint8)).resize((self.image_size, self.image_size))
            image = np.array(image, dtype=np.float32)
            mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((self.image_size, self.image_size))
            mask = np.array(mask, dtype=np.float32) / 255.0
        
        # Normalize image to [0, 1]
        if self.normalize:
            image = image / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


def get_augmentation(image_size=256, augmentation_prob=0.5):
    """Get augmentation pipeline."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=augmentation_prob),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=augmentation_prob),
            A.GaussNoise(p=0.2),
            A.OneOf(
                [
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.CLAHE(p=1.0),
                    A.RandomBrightnessContrast(p=1.0),
                ],
                p=0.2,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=None,
    )


def get_validation_augmentation(image_size=256):
    """Get validation augmentation pipeline (no aggressive augmentation)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    )


def create_data_loaders(
    train_images_dir,
    train_masks_dir,
    val_images_dir,
    val_masks_dir,
    batch_size=16,
    image_size=256,
    num_workers=4,
    pin_memory=True,
    augmentation_prob=0.5,
):
    """
    Create train and validation data loaders.
    
    Args:
        train_images_dir: Path to training images
        train_masks_dir: Path to training masks
        val_images_dir: Path to validation images
        val_masks_dir: Path to validation masks
        batch_size: Batch size
        image_size: Image size to resize to
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        augmentation_prob: Probability for augmentations
    
    Returns:
        train_loader, val_loader
    """
    # Training augmentation
    train_aug = get_augmentation(image_size, augmentation_prob)
    train_dataset = CellSegmentationDataset(
        train_images_dir,
        train_masks_dir,
        image_size=image_size,
        augmentation=train_aug,
        normalize=False,
    )
    
    # Validation augmentation (minimal)
    val_aug = get_validation_augmentation(image_size)
    val_dataset = CellSegmentationDataset(
        val_images_dir,
        val_masks_dir,
        image_size=image_size,
        augmentation=val_aug,
        normalize=False,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader
