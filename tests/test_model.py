

import unittest
import os
import torch
import tempfile

from scripts.model import Code2VecBugHunter, PathEncoder


class TestModel(unittest.TestCase):
    """Test cases for the Code2Vec model"""
    
    def setUp(self):
        """Set up test environment"""
        self.path_vocab_size = 100
        self.embedding_dim = 64
        self.hidden_dim = 64
        self.num_layers = 2
        
        # Create model for testing
        self.model = Code2VecBugHunter(
            path_vocab_size=self.path_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
    
    def test_path_encoder(self):
        """Test path encoder module"""
        encoder = PathEncoder(
            path_vocab_size=self.path_vocab_size,
            embedding_dim=self.embedding_dim
        )
        
        # Test with batch of 2 samples, 5 paths each
        paths = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        
        # Forward pass
        embeddings = encoder(paths)
        
        # Check output shape
        self.assertEqual(embeddings.shape, (2, 5, self.embedding_dim))
    
    def test_model_forward(self):
        """Test model forward pass"""
        # Create input batch
        # Batch of 3 samples, 10 paths each
        paths = torch.randint(0, self.path_vocab_size, (3, 10))
        
        # Forward pass
        outputs = self.model(paths)
        
        # Check output shapes
        self.assertEqual(outputs['logits'].shape, (3,))
        self.assertEqual(outputs['attention_weights'].shape, (3, 10))
    
    def test_model_predict(self):
        """Test model prediction"""
        # Create input batch
        paths = torch.randint(0, self.path_vocab_size, (2, 10))
        
        # Get predictions
        predictions = self.model.predict(paths)
        
        # Check output shapes and types
        self.assertEqual(predictions['is_buggy'].shape, (2,))
        self.assertEqual(predictions['confidence'].shape, (2,))
        self.assertEqual(predictions['attention_weights'].shape, (2, 10))
        
        # Check value ranges
        self.assertTrue(torch.all((predictions['confidence'] >= 0) & (predictions['confidence'] <= 1)))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create temporary path for model
        with tempfile.NamedTemporaryFile() as tmp:
            model_path = tmp.name
            
            # Save model
            self.model.save(model_path)
            
            # Load model
            loaded_model = Code2VecBugHunter.load(model_path)
            
            # Check model parameters
            self.assertEqual(
                loaded_model.encoder.path_embedding.num_embeddings,
                self.model.encoder.path_embedding.num_embeddings
            )
            self.assertEqual(
                loaded_model.encoder.path_embedding.embedding_dim,
                self.model.encoder.path_embedding.embedding_dim
            )
            
            # Create test input
            paths = torch.randint(0, self.path_vocab_size, (1, 10))
            
            # Compare outputs
            self.model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                original_output = self.model(paths)
                loaded_output = loaded_model(paths)
            
            # Verify that the outputs are equal
            torch.testing.assert_close(
                original_output['logits'],
                loaded_output['logits']
            )


if __name__ == '__main__':
    unittest.main()
