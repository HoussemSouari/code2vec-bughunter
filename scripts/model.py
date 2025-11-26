

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


class PathEncoder(nn.Module):
    """
    Encoder for AST paths and associated tokens
    """
    def __init__(self, 
                 path_vocab_size: int,
                 token_vocab_size: int = 0,  # Optional for simplified version
                 embedding_dim: int = 128,
                 dropout: float = 0.1):
        """
        Initialize the path encoder
        
        Args:
            path_vocab_size: Size of path vocabulary
            token_vocab_size: Size of token vocabulary (optional)
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
        """
        super(PathEncoder, self).__init__()
        
        self.path_embedding = nn.Embedding(
            path_vocab_size, embedding_dim, padding_idx=0
        )
        
        # In a complete implementation, we'd also have token embeddings
        # and combine them with path embeddings
        self.token_embedding = None
        if token_vocab_size > 0:
            self.token_embedding = nn.Embedding(
                token_vocab_size, embedding_dim, padding_idx=0
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through path encoder
        
        Args:
            paths: Tensor of path indices [batch_size, max_paths]
            
        Returns:
            Path embeddings [batch_size, max_paths, embedding_dim]
        """
        # [batch_size, max_paths, embedding_dim]
        path_embeddings = self.path_embedding(paths)

        # If token embeddings are available, this method will be called
        # with start_ids and end_ids by the higher-level encoder forward.
        return self.dropout(path_embeddings)


class Code2VecBugHunter(nn.Module):
    """
    Code2Vec-based model for bug detection
    """
    def __init__(self, 
                 path_vocab_size: int,
                 token_vocab_size: int = 0,
                 embedding_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        """
        Initialize the model
        
        Args:
            path_vocab_size: Size of path vocabulary
            token_vocab_size: Size of token vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of layers in the network
            dropout: Dropout rate
        """
        super(Code2VecBugHunter, self).__init__()
        
        self.encoder = PathEncoder(
            path_vocab_size=path_vocab_size,
            token_vocab_size=token_vocab_size,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Linear(embedding_dim, 1)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim
        
        for i in range(num_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Output layer for binary classification
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, paths: torch.Tensor, start_ids: Optional[torch.Tensor] = None, end_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            paths: Tensor of path indices [batch_size, max_paths]
            
        Returns:
            Dictionary with model outputs including:
            - logits: Raw prediction scores
            - attention_weights: Attention weights for each path
        """
        # Get path embeddings [batch_size, max_paths, embedding_dim]
        path_embeddings = self.encoder(paths)

        # If token embeddings exist and token ids are provided, combine them
        if self.encoder.token_embedding is not None and start_ids is not None and end_ids is not None:
            # [batch, max_paths, dim]
            start_emb = self.encoder.token_embedding(start_ids)
            end_emb = self.encoder.token_embedding(end_ids)
            # Combine embeddings (sum)
            path_embeddings = path_embeddings + start_emb + end_emb

        # Calculate attention scores [batch_size, max_paths, 1]
        attention_scores = self.attention(path_embeddings)

        # Squeeze to [batch, max_paths]
        attention_logits = attention_scores.squeeze(-1)

        # If mask provided, set padded positions to large negative value before softmax
        if mask is not None:
            # mask: 1 for valid, 0 for pad
            # Create boolean mask
            bool_mask = (mask == 1)
            # Where mask is False, set logits to a large negative value
            attention_logits = attention_logits.masked_fill(~bool_mask, float('-1e9'))

        # Apply softmax to get attention weights [batch_size, max_paths]
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Apply attention to get code vector [batch_size, embedding_dim]
        # We use unsqueeze to add a dimension for broadcasting
        # [batch_size, 1, max_paths] * [batch_size, max_paths, embedding_dim]
        code_vectors = torch.bmm(
            attention_weights.unsqueeze(1),
            path_embeddings
        ).squeeze(1)
        
        # Pass through fully connected layers
        x = code_vectors
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
            x = self.dropout(x)
        
        # Get final logits
        logits = self.output_layer(x)
        
        return {
            'logits': logits.squeeze(-1),
            'attention_weights': attention_weights
        }
    
    def predict(self, paths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make a prediction with the model
        
        Args:
            paths: Tensor of path indices [batch_size, max_paths]
            
        Returns:
            Dictionary with prediction results:
            - is_buggy: Boolean prediction
            - confidence: Confidence score
            - attention: Attention weights
        """
        self.eval()
        with torch.no_grad():
            # Support optional start/end ids and mask if passed in a tuple
            if isinstance(paths, (list, tuple)):
                # Expect (paths, start_ids, end_ids, mask)
                outputs = self.forward(*paths)
            else:
                outputs = self.forward(paths)
            
            # Apply sigmoid to get probability
            probs = torch.sigmoid(outputs['logits'])
            
            # Make binary prediction
            predictions = (probs >= 0.5).float()
            
            return {
                'is_buggy': predictions,
                'confidence': probs,
                'attention_weights': outputs['attention_weights']
            }
    
    def save(self, path: str):
        """Save model to disk"""
        model_state = {
            'path_vocab_size': self.encoder.path_embedding.num_embeddings,
            'embedding_dim': self.encoder.path_embedding.embedding_dim,
            'hidden_dim': self.fc_layers[0].out_features if self.fc_layers else 0,
            'num_layers': len(self.fc_layers),
            'state_dict': self.state_dict()
        }
        # Optionally include vocabularies if attached to the model
        if hasattr(self, 'path_vocab') and self.path_vocab:
            model_state['path_vocab'] = self.path_vocab
        if hasattr(self, 'token_vocab') and self.token_vocab:
            model_state['token_vocab'] = self.token_vocab

        torch.save(model_state, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'Code2VecBugHunter':
        """Load model from disk"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model_state = torch.load(path, map_location=device)

        token_vocab = model_state.get('token_vocab', {})
        token_vocab_size = len(token_vocab) + 1 if token_vocab else 0

        # Create model with saved parameters
        model = cls(
            path_vocab_size=model_state['path_vocab_size'],
            token_vocab_size=token_vocab_size,
            embedding_dim=model_state['embedding_dim'],
            hidden_dim=model_state['hidden_dim'],
            num_layers=model_state['num_layers']
        )
        
        # Load weights
        model.load_state_dict(model_state['state_dict'])
        # Attach vocabs if present
        model.path_vocab = model_state.get('path_vocab', {})
        model.token_vocab = model_state.get('token_vocab', {})
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model
