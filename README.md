# Code2Vec-BugHunter: Deep Learning for Bug Detection

A Python-based deep learning system that creates code embeddings to detect bugs in source code using a Code2Vec-inspired approach.

## Overview

Code2Vec-BugHunter is a deep learning system that analyzes Python code to detect potential bugs. The system leverages the Code2Vec approach to convert source code into vector embeddings that capture semantic information about the code structure. These embeddings are then used to classify whether a code snippet contains bugs.

## Features

- **AST Parsing**: Parses Python source code into abstract syntax trees (ASTs)
- **Path Extraction**: Converts ASTs into path-based representations (Code2Vec approach)
- **Neural Network Model**: Trains a deep learning model to generate code embeddings
- **Bug Detection**: Classifies code snippets as buggy or non-buggy
- **Visualization**: Provides visualization of model attention/feature importance
- **Web Demo**: Includes a web interface for easy interaction
- **Command Line Interface**: Supports CLI for training and inference

## System Architecture

The system follows a multi-stage pipeline:

1. **Code Parsing**: Source code is parsed into an Abstract Syntax Tree (AST)
2. **Path Extraction**: AST paths are extracted to capture code structure
3. **Embedding Generation**: A neural network creates vector embeddings from code paths
4. **Bug Classification**: Model predicts if code contains bugs based on embeddings
5. **Attention Visualization**: Highlights suspicious code elements

<div align="center">
    <img src="https://via.placeholder.com/800x400.png?text=Code2Vec-BugHunter+Architecture" alt="Architecture Diagram" width="800"/>
</div>

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NetworkX
- Flask (for web demo)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code2vec-bughunter.git
cd code2vec-bughunter
