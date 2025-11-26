
import os
import logging
import torch
from typing import Dict, List, Tuple, Any

from scripts.model import Code2VecBugHunter
from utils.ast_utils import normalize_and_parse_code, extract_ast_paths
from utils.visualization import visualize_attention
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeInference:
    """Handles inference on code snippets using a trained model"""
    
    def __init__(self, model_path: str, max_paths: int = 100, max_path_length: int = 8):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the trained model
            max_paths: Maximum number of paths to use per sample
            max_path_length: Maximum length of paths to extract
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        
        # Load model
        try:
            self.model = Code2VecBugHunter.load(model_path, self.device)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Simplified placeholder for vocabulary
        # In a real implementation, this would be loaded from the training artifacts
        # Load vocabularies from the model checkpoint if available
        self.path_vocab = getattr(self.model, 'path_vocab', {}) or {}
        self.token_vocab = getattr(self.model, 'token_vocab', {}) or {}
        
    def _preprocess_code(self, code: str) -> Dict:
        """
        Preprocess code snippet for inference
        
        Args:
            code: Source code string
            
        Returns:
            Dictionary with preprocessed features
        """
        try:
            # Parse AST and extract paths
            ast_tree = normalize_and_parse_code(code)
            path_contexts = extract_ast_paths(ast_tree, self.max_paths, self.max_path_length)
            
            # Convert paths and tokens to indices using saved vocabularies
            path_indices = torch.zeros(self.max_paths, dtype=torch.long)
            start_indices = torch.zeros(self.max_paths, dtype=torch.long)
            end_indices = torch.zeros(self.max_paths, dtype=torch.long)

            for i, path_context in enumerate(path_contexts[:self.max_paths]):
                # Use 0 as padding/unknown index (embedding padding_idx=0)
                path_idx = self.path_vocab.get(path_context['path'], 0)
                start_idx = self.token_vocab.get(path_context.get('start_token', ''), 0)
                end_idx = self.token_vocab.get(path_context.get('end_token', ''), 0)

                path_indices[i] = path_idx
                start_indices[i] = start_idx
                end_indices[i] = end_idx

            mask = (path_indices != 0).long()

            return {
                'paths': path_indices.unsqueeze(0).to(self.device),  # Add batch dim
                'start_ids': start_indices.unsqueeze(0).to(self.device),
                'end_ids': end_indices.unsqueeze(0).to(self.device),
                'mask': mask.unsqueeze(0).to(self.device),
                'path_contexts': path_contexts,
                'code': code
            }
        except SyntaxError:
            logger.error("Syntax error in code")
            raise
        except Exception as e:
            logger.error(f"Error preprocessing code: {e}")
            raise
    
    def predict(self, code: str) -> Dict:
        """
        Make a prediction on a code snippet
        
        Args:
            code: Source code string
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Preprocess code
        preprocessed = self._preprocess_code(code)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model.predict((preprocessed['paths'], preprocessed['start_ids'], preprocessed['end_ids'], preprocessed['mask']))
            
            # Extract results
            is_buggy = bool(outputs['is_buggy'][0].item())
            confidence = float(outputs['confidence'][0].item())
            attention_weights = outputs['attention_weights'][0].cpu().numpy()
            
            # Map attention weights to path contexts
            path_attention = []
            for i, weight in enumerate(attention_weights[:len(preprocessed['path_contexts'])]):
                path_context = preprocessed['path_contexts'][i]
                path_str = f"{path_context['start_token']} -> {path_context['path']} -> {path_context['end_token']}"
                path_attention.append((path_str, float(weight)))
            
            # Sort by attention weight
            path_attention.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'is_buggy': is_buggy,
                'confidence': confidence,
                'attention': path_attention,
                'code': code
            }


def run_inference(model_path: str, code: str) -> Dict:
    """
    Run inference on a code snippet
    
    Args:
        model_path: Path to the trained model
        code: Source code string
        
    Returns:
        Dictionary with prediction results
    """
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Check if model exists, if not create a dummy model for demonstration
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}, creating a dummy model")
        _create_dummy_model(model_path)
    
    # Initialize inference engine
    inference = CodeInference(model_path)
    
    # Make prediction
    result = inference.predict(code)
    
    # Generate visualization
    vis_path = os.path.join(os.path.dirname(model_path), 'attention_visualization.html')
    visualize_attention(result['code'], result['attention'], vis_path)
    
    return result


def _create_dummy_model(model_path: str):
    """
    Create a dummy model for demonstration purposes
    
    Args:
        model_path: Path to save the dummy model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create a simple model
    model = Code2VecBugHunter(
        path_vocab_size=100,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2
    )
    
    # Save the model
    model.save(model_path)
    

if __name__ == "__main__":
    # Example standalone usage
    logging.basicConfig(level=logging.INFO)
    
    code_sample = """
def get_element(arr, index):
    return arr[index]  # Missing bounds check
"""
    
    result = run_inference('models/code2vec_bughunter.pt', code_sample)
    
    print(f"Bug Detection Result:")
    print(f"  - Buggy: {'Yes' if result['is_buggy'] else 'No'}")
    print(f"  - Confidence: {result['confidence']:.4f}")
    
    if 'attention' in result:
        print(f"  - Top attention areas:")
        for i, (node, weight) in enumerate(result['attention'][:5], 1):
            print(f"    {i}. {node}: {weight:.4f}")
