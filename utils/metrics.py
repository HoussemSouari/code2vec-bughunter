"""
Metrics utilities for model evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(labels: Union[List, np.ndarray], 
                   predictions: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        labels: True labels
        predictions: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy arrays if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    
    # For binary classification
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def compute_confusion_matrix(labels: Union[List, np.ndarray], 
                            predictions: Union[List, np.ndarray]) -> Dict[str, int]:
    """
    Compute confusion matrix
    
    Args:
        labels: True labels
        predictions: Predicted labels
        
    Returns:
        Dictionary with confusion matrix elements
    """
    # Convert to numpy arrays if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Compute true positives, false positives, true negatives, false negatives
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }
