"""
Web demonstration interface for Code2Vec-BugHunter.
"""

import os
import logging
import json
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch

from inference import CodeInference
from utils.visualization import visualize_attention

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_for_testing")

# Global variables
inference_engine = None


def init_inference_engine(model_path: str):
    """Initialize the inference engine with the model"""
    global inference_engine
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}. Creating dummy model.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create empty model file for demonstration if it doesn't exist
        from model import Code2VecBugHunter
        model = Code2VecBugHunter(
            path_vocab_size=100,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2
        )
        model.save(model_path)
    
    # Initialize inference engine
    inference_engine = CodeInference(model_path)
    logger.info(f"Inference engine initialized with model from {model_path}")


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/demo')
def demo():
    """Demo page"""
    return render_template('demo.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    """API endpoint to analyze code"""
    code = request.form.get('code', '')
    
    if not code:
        return jsonify({
            'success': False,
            'error': 'No code provided'
        })
    
    try:
        # Run inference
        result = inference_engine.predict(code)
        
        # Generate visualization
        visualization_path = os.path.join('static', 'visualizations', 'latest.html')
        os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
        
        visualize_attention(result['code'], result['attention'], visualization_path)
        
        # Return results
        return jsonify({
            'success': True,
            'is_buggy': result['is_buggy'],
            'confidence': result['confidence'],
            'attention': result['attention'][:10],  # Return top 10 attention areas
            'visualization_url': url_for('static', filename='visualizations/latest.html')
        })
    
    except SyntaxError as e:
        return jsonify({
            'success': False,
            'error': f'Syntax error in the code: {str(e)}'
        })
    
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        return jsonify({
            'success': False,
            'error': f'Error analyzing code: {str(e)}'
        })


def start_web_server(model_path: str, port: int = 5000):
    """Start the web server"""
    init_inference_engine(model_path)
    
    # Create static directories
    os.makedirs(os.path.join('static', 'visualizations'), exist_ok=True)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Start the web server
    start_web_server('models/code2vec_bughunter.pt')
