"""Evaluation metrics for segmentation."""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


class SegmentationMetrics:
    """Calculate segmentation metrics."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
    
    def update(self, predictions, targets):
        """
        Update metrics with batch.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks
        """
        # Convert to probabilities
        probs = torch.sigmoid(predictions).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Apply threshold
        pred_binary = (probs > self.threshold).astype(np.float32)
        
        # Flatten
        pred_binary = pred_binary.flatten()
        targets = targets.flatten()
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, pred_binary).ravel()
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
    
    def get_metrics(self):
        """Get all metrics."""
        eps = 1e-7
        
        # Sensitivity (Recall)
        sensitivity = self.tp / (self.tp + self.fn + eps)
        
        # Specificity
        specificity = self.tn / (self.tn + self.fp + eps)
        
        # Precision
        precision = self.tp / (self.tp + self.fp + eps)
        
        # F1 Score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
        
        # Dice coefficient
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        
        # Jaccard/IoU
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        
        # Accuracy
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy,
        }


def calculate_dice(predictions, targets, threshold=0.5):
    """Calculate Dice coefficient."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs > threshold).float()
    
    intersection = (pred_binary * targets).sum()
    union = pred_binary.sum() + targets.sum()
    
    dice = 2.0 * intersection / (union + 1e-7)
    return dice.item()


def calculate_iou(predictions, targets, threshold=0.5):
    """Calculate Intersection over Union (IoU)."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs > threshold).float()
    
    intersection = (pred_binary * targets).sum()
    union = (pred_binary + targets - pred_binary * targets).sum()
    
    iou = intersection / (union + 1e-7)
    return iou.item()


def calculate_sensitivity(predictions, targets, threshold=0.5):
    """Calculate Sensitivity (Recall/True Positive Rate)."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs > threshold).float()
    
    tp = (pred_binary * targets).sum()
    fn = ((1 - pred_binary) * targets).sum()
    
    sensitivity = tp / (tp + fn + 1e-7)
    return sensitivity.item()


def calculate_specificity(predictions, targets, threshold=0.5):
    """Calculate Specificity (True Negative Rate)."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs > threshold).float()
    
    tn = ((1 - pred_binary) * (1 - targets)).sum()
    fp = (pred_binary * (1 - targets)).sum()
    
    specificity = tn / (tn + fp + 1e-7)
    return specificity.item()


def calculate_precision(predictions, targets, threshold=0.5):
    """Calculate Precision."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs > threshold).float()
    
    tp = (pred_binary * targets).sum()
    fp = (pred_binary * (1 - targets)).sum()
    
    precision = tp / (tp + fp + 1e-7)
    return precision.item()


def calculate_f1(predictions, targets, threshold=0.5):
    """Calculate F1 Score."""
    precision = calculate_precision(predictions, targets, threshold)
    sensitivity = calculate_sensitivity(predictions, targets, threshold)
    
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-7)
    return f1


def calculate_auc(predictions, targets):
    """Calculate AUC-ROC."""
    probs = torch.sigmoid(predictions).detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    
    try:
        auc = roc_auc_score(targets, probs)
    except:
        auc = 0.0
    
    return auc
