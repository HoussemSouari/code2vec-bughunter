
import argparse
import logging
import os
import sys

from scripts.train import train_model
from scripts.inference import run_inference
from web_demo import app, init_inference_engine

# Configure logging for the application
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize the inference engine with a default model path for web server
model_path = 'models/code2vec_bughunter.pt'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
init_inference_engine(model_path)

# Create static directories for visualizations
os.makedirs('static/visualizations', exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Code2Vec-BugHunter: Deep Learning for Bug Detection'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data-path', type=str, default='data',
                             help='Path to the dataset')
    train_parser.add_argument('--model-path', type=str, default='models/code2vec_bughunter.pt',
                             help='Path to save the trained model')
    train_parser.add_argument('--epochs', type=int, default=10,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate for optimizer')
    train_parser.add_argument('--embedding-dim', type=int, default=128,
                             help='Dimension of code embeddings')
    
    # Inference subcommand
    inference_parser = subparsers.add_parser('inference', help='Run inference on code')
    inference_parser.add_argument('--model-path', type=str, required=True,
                                 help='Path to the trained model')
    inference_parser.add_argument('--file', type=str, help='Path to the Python file for inference')
    inference_parser.add_argument('--code', type=str, help='Python code string for inference')
    
    # Web demo subcommand
    web_parser = subparsers.add_parser('web', help='Start web demo interface')
    web_parser.add_argument('--model-path', type=str, default='models/code2vec_bughunter.pt',
                           help='Path to the trained model for inference')
    web_parser.add_argument('--port', type=int, default=5000,
                           help='Port to run the web server on')
    
    return parser.parse_args()


def main():
    """Main function to handle CLI commands"""
    args = parse_arguments()
    
    if args.command == 'train':
        # Create the model directory if it doesn't exist
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
        train_model(
            data_path=args.data_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            embedding_dim=args.embedding_dim
        )
    
    elif args.command == 'inference':
        if not (args.file or args.code):
            print("Error: Either --file or --code must be provided")
            sys.exit(1)
            
        if args.file:
            with open(args.file, 'r') as f:
                code = f.read()
        else:
            code = args.code
            
        result = run_inference(args.model_path, code)
        
        print(f"Bug Detection Result:")
        print(f"  - Buggy: {'Yes' if result['is_buggy'] else 'No'}")
        print(f"  - Confidence: {result['confidence']:.4f}")
        
        if 'attention' in result:
            print(f"  - Top attention areas:")
            for i, (node, weight) in enumerate(result['attention'][:5], 1):
                print(f"    {i}. {node}: {weight:.4f}")
    
    elif args.command == 'web':
        # For manual runs, not through gunicorn
        app.run(host='0.0.0.0', port=args.port, debug=True)
    
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
