"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss (F1 score)."""
    
    def __init__(self, smooth=1.0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Calculate Dice loss.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth masks
        
        Returns:
            Dice loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combination of Binary Cross Entropy and Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, predictions, targets):
        """
        Calculate combined BCE + Dice loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
        
        Returns:
            Combined loss value
        """
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Calculate Focal loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        probs = torch.sigmoid(predictions)
        
        # Calculate BCE
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Calculate focal loss
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * ((1 - p_t) ** self.gamma) * bce
        
        return focal.mean()


class JaccardLoss(nn.Module):
    """Jaccard/IoU loss."""
    
    def __init__(self, smooth=1.0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Calculate Jaccard loss.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth masks
        
        Returns:
            Jaccard loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        # Calculate Jaccard
        jaccard = (intersection + self.smooth) / (union + self.smooth + self.eps)
        
        return 1.0 - jaccard


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced segmentation."""
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Calculate Tversky loss.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth masks
        
        Returns:
            Tversky loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate components
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps)
        
        return 1.0 - tversky


def get_loss_function(loss_name="bce_dice", **kwargs):
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function ('bce', 'dice', 'bce_dice', 'focal', 'jaccard', 'tversky')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_functions = {
        'bce': nn.BCEWithLogitsLoss,
        'dice': DiceLoss,
        'bce_dice': BCEDiceLoss,
        'focal': FocalLoss,
        'jaccard': JaccardLoss,
        'tversky': TverskyLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
