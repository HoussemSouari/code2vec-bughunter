"""
Training module for Code2Vec-BugHunter.
Handles model training, evaluation, and checkpointing.
"""

import os
import logging
import time
from typing import Dict, Tuple, List, Optional
import json
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Code2VecBugHunter
from data_loader import DataManager
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train_model(data_path: str,
               model_path: str,
               epochs: int = 10,
               batch_size: int = 64,
               learning_rate: float = 0.001,
               embedding_dim: int = 128) -> Code2VecBugHunter:
    """
    Train the Code2Vec model for bug detection
    
    Args:
        data_path: Path to the dataset
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        embedding_dim: Dimension of code embeddings
        
    Returns:
        Trained model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    data_manager = DataManager()
    dataset_path = data_manager.download_dataset('defects4j_subset')
    train_loader, val_loader, test_loader = data_manager.load_datasets(dataset_path)
    
    # Initialize model
    path_vocab_size = data_manager.get_vocab_size()
    
    model = Code2VecBugHunter(
        path_vocab_size=path_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=embedding_dim,
        num_layers=2,
        dropout=0.1
    )
    
    model.to(device)
    logger.info(f"Initialized model with path_vocab_size={path_vocab_size}")
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in train_loader:
            paths = batch['paths'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(paths)
            logits = outputs['logits']
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        train_loss /= train_steps
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                paths = batch['paths'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(paths)
                logits = outputs['logits']
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_steps += 1
        
        val_loss /= val_steps
        val_losses.append(val_loss)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        val_f1 = metrics['f1']
        val_f1_scores.append(val_f1)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val F1: {val_f1:.4f}, "
                   f"Val Precision: {metrics['precision']:.4f}, "
                   f"Val Recall: {metrics['recall']:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save(model_path)
            logger.info(f"Saved new best model with Val F1: {val_f1:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_f1_scores)
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    evaluate_model(model, test_loader, device, criterion)
    
    # Save final model if a best model wasn't saved
    if best_val_f1 == 0.0:
        model.save(model_path)
        logger.info(f"Saved final model")
    
    return model


def evaluate_model(model: Code2VecBugHunter, 
                  data_loader: DataLoader,
                  device: torch.device,
                  criterion: nn.Module) -> Dict:
    """
    Evaluate the model on a dataset
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run on
        criterion: Loss function
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    all_preds = []
    all_labels = []
    all_attention = []
    
    with torch.no_grad():
        for batch in data_loader:
            paths = batch['paths'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(paths)
            logits = outputs['logits']
            attention_weights = outputs['attention_weights']
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention.extend(attention_weights.cpu().numpy())
            
            steps += 1
    
    avg_loss = total_loss / steps
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    logger.info(f"Evaluation - "
               f"Loss: {avg_loss:.4f}, "
               f"Accuracy: {metrics['accuracy']:.4f}, "
               f"F1: {metrics['f1']:.4f}, "
               f"Precision: {metrics['precision']:.4f}, "
               f"Recall: {metrics['recall']:.4f}")
    
    return metrics


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float], 
                        val_f1_scores: List[float],
                        save_path: str = 'training_curves.png'):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_f1_scores: List of validation F1 scores
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation F1 score
    plt.subplot(2, 1, 2)
    plt.plot(val_f1_scores, label='Validation F1 Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    # Example standalone usage
    logging.basicConfig(level=logging.INFO)
    train_model(
        data_path='data',
        model_path='models/code2vec_bughunter.pt',
        epochs=5
    )
