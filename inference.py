"""Inference script for U-Net cell segmentation model."""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.unet import create_unet
from src.utils import get_device, load_model


class SegmentationInference:
    """Inference class for segmentation."""
    
    def __init__(self, model_path, device='cuda', image_size=256, threshold=0.5):
        """
        Initialize inference.
        
        Args:
            model_path: Path to trained model
            device: Device to use
            image_size: Size to resize images to
            threshold: Confidence threshold for binarization
        """
        self.device = get_device(device)
        self.image_size = image_size
        self.threshold = threshold
        
        # Create and load model
        self.model = create_unet(
            in_channels=3,
            out_channels=1,
            init_features=64,
            depth=4,
        ).to(self.device)
        
        load_model(self.model, model_path, device=str(self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def preprocess(self, image_path):
        """
        Preprocess image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Preprocessed image tensor, original size
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size  # (width, height)
        
        # Resize
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return image, orig_size
    
    def postprocess(self, mask, original_size=None):
        """
        Postprocess mask.
        
        Args:
            mask: Model output mask
            original_size: Original image size (width, height)
        
        Returns:
            Binary mask
        """
        # Convert to probability
        mask = torch.sigmoid(mask)
        mask = mask.squeeze().detach().cpu().numpy()
        
        # Apply threshold
        binary_mask = (mask > self.threshold).astype(np.uint8) * 255
        
        # Resize to original size if provided
        if original_size is not None:
            binary_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_LINEAR)
            binary_mask = (binary_mask > 127.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def segment(self, image_path, output_path=None):
        """
        Segment image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output mask
        
        Returns:
            Binary mask
        """
        # Preprocess
        image, orig_size = self.preprocess(image_path)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image)
        
        # Postprocess
        mask = self.postprocess(logits, original_size=orig_size)
        
        # Save output if path provided
        if output_path:
            Image.fromarray(mask).save(output_path)
            print(f"Mask saved to {output_path}")
        
        return mask
    
    def segment_batch(self, image_dir, output_dir=None):
        """
        Segment batch of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save output masks
        
        Returns:
            List of masks
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir) if output_dir else None
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_files = sorted([f for f in image_dir.iterdir() 
                            if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        
        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            return []
        
        masks = []
        for image_path in tqdm(image_files, desc="Segmenting"):
            # Segment
            mask = self.segment(str(image_path))
            masks.append(mask)
            
            # Save output
            if output_dir:
                output_path = output_dir / image_path.name.replace(image_path.suffix, '_mask.png')
                Image.fromarray(mask).save(output_path)
        
        return masks
    
    def segment_with_visualization(self, image_path, output_path=None, alpha=0.5):
        """
        Segment image and create visualization.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            alpha: Blending alpha for overlay
        
        Returns:
            Visualization image
        """
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Segment
        mask = self.segment(image_path)
        
        # Resize mask to match image size
        image_h, image_w = image.shape[:2]
        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 127] = [0, 255, 0]  # Green for segmented regions
        
        # Blend
        visualization = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        # Save
        if output_path:
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), visualization_bgr)
            print(f"Visualization saved to {output_path}")
        
        return visualization


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Inference with U-Net segmentation model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image-dir', type=str, help='Directory with images')
    parser.add_argument('--output', type=str, help='Path to save output mask')
    parser.add_argument('--output-dir', type=str, help='Directory to save output masks')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error('Either --image or --image-dir must be provided')
    
    # Create inferencer
    inferencer = SegmentationInference(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold,
    )
    
    # Single image
    if args.image:
        print(f"Segmenting {args.image}")
        mask = inferencer.segment(args.image, output_path=args.output)
        
        if args.visualize:
            viz_path = args.output.replace('.png', '_viz.png') if args.output else 'visualization.png'
            inferencer.segment_with_visualization(args.image, output_path=viz_path)
    
    # Batch
    elif args.image_dir:
        print(f"Segmenting images in {args.image_dir}")
        masks = inferencer.segment_batch(args.image_dir, output_dir=args.output_dir)
        print(f"Segmented {len(masks)} images")


if __name__ == '__main__':
    main()
