
import os
import logging
import random
import json
import ast
import pickle
from typing import Dict, List, Tuple, Set, Generator, Optional, Any
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import shutil

import torch
from torch.utils.data import Dataset, DataLoader

from utils.ast_utils import extract_ast_paths, normalize_and_parse_code


logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset class for code snippets with buggy/non-buggy labels"""
    
    def __init__(self, 
                 samples: List[Dict], 
                 path_vocab: Dict[str, int], 
                 token_vocab: Dict[str, int] = None,
                 max_paths: int = 100, 
                 max_path_length: int = 8):
        """
        Initialize the dataset
        
        Args:
            samples: List of code samples with their metadata
            path_vocab: Vocabulary mapping path contexts to indices
            max_paths: Maximum number of paths to keep per sample
            max_path_length: Maximum length of path contexts
        """
        self.samples = samples
        self.path_vocab = path_vocab
        self.token_vocab = token_vocab or {}
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        self.pad_token = 0  # Use 0 as padding token
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single code sample with its processed features"""
        sample = self.samples[idx]
        
        # Extract paths, start_tokens and end_tokens
        paths_list = []
        for path_context in sample['path_contexts'][:self.max_paths]:
            path_idx = self.path_vocab.get(path_context['path'], self.pad_token)
            start_token = path_context['start_token']
            end_token = path_context['end_token']
            paths_list.append((path_idx, start_token, end_token))
        
        # Pad if needed
        while len(paths_list) < self.max_paths:
            paths_list.append((self.pad_token, '', ''))
        
        # Convert to tensors: path indices, start token ids, end token ids
        paths_tensor = torch.tensor([p[0] for p in paths_list], dtype=torch.long)

        # Map tokens to ids using token_vocab (0 = padding / unknown)
        start_ids = [self.token_vocab.get(p[1], self.pad_token) if p[1] else self.pad_token for p in paths_list]
        end_ids = [self.token_vocab.get(p[2], self.pad_token) if p[2] else self.pad_token for p in paths_list]

        start_tensor = torch.tensor(start_ids, dtype=torch.long)
        end_tensor = torch.tensor(end_ids, dtype=torch.long)

        # Mask to indicate real paths (1) vs padding (0)
        mask = (paths_tensor != self.pad_token).long()

        return {
            'paths': paths_tensor,
            'start_ids': start_tensor,
            'end_ids': end_tensor,
            'mask': mask,
            'label': torch.tensor(sample['is_buggy'], dtype=torch.float)
        }


class DataManager:
    """Handles data downloading, processing and loading"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data manager
        
        Args:
            data_dir: Directory to store/load datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Default dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.path_vocab = {}
        self.token_vocab = {}
        
        # Metadata
        self.max_paths = 100
        self.max_path_length = 8
        self.max_token_length = 5
        
    def download_dataset(self, dataset_name: str = 'defects4j_subset') -> str:
        """
        Download a dataset for bug detection. Currently supported:
        - defects4j_subset: A small subset of the Defects4J dataset
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            Path to the downloaded dataset
        """
        dataset_dir = self.data_dir / dataset_name
        
        if dataset_dir.exists():
            logger.info(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return str(dataset_dir)
        
        dataset_dir.mkdir(exist_ok=True)
        
        if dataset_name == 'defects4j_subset':
            # In a real implementation, this would download from a real source
            # For this implementation, we'll create synthetic data
            logger.info(f"Creating synthetic dataset for {dataset_name}")
            
            # Create a synthetic dataset with 100 samples
            # In a real implementation, you would download from a URL:
            # url = "https://example.com/datasets/defects4j_subset.tar.gz"
            # urllib.request.urlretrieve(url, self.data_dir / "defects4j_subset.tar.gz")
            
            self._create_synthetic_dataset(dataset_dir)
            
            logger.info(f"Created synthetic dataset at {dataset_dir}")
            return str(dataset_dir)
        elif dataset_name == 'bugs_in_py':
            logger.info("Preparing BugsInPy dataset (requires 'bugsinpy' package)")
            os.makedirs(dataset_dir / 'train', exist_ok=True)
            os.makedirs(dataset_dir / 'val', exist_ok=True)
            os.makedirs(dataset_dir / 'test', exist_ok=True)

            # Try to download and prepare BugsInPy-derived dataset
            success = self._download_and_prepare_bugsinpy(dataset_dir, max_bugs=500)
            if not success:
                # Fallback: create small example layout to keep API usable
                logger.warning("BugsInPy extraction failed; creating example files. Install 'bugsinpy' for full dataset support.")
                example = [{
                    "id": "example_1",
                    "code": "def get_element(arr, index):\n    return arr[index]  # Missing bounds check",
                    "is_buggy": True
                }, {
                    "id": "example_2",
                    "code": "def get_element(arr, index):\n    if 0 <= index < len(arr):\n        return arr[index]\n    return None",
                    "is_buggy": False
                }]

                with open(dataset_dir / 'train' / 'samples.json', 'w') as f:
                    json.dump(example, f, indent=2)
                with open(dataset_dir / 'val' / 'samples.json', 'w') as f:
                    json.dump(example[:1], f, indent=2)
                with open(dataset_dir / 'test' / 'samples.json', 'w') as f:
                    json.dump(example[1:], f, indent=2)

                return str(dataset_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _create_synthetic_dataset(self, dataset_dir: Path):
        """
        Create a synthetic dataset for demonstration purposes
        
        Args:
            dataset_dir: Directory to store the dataset
        """
        # Create train/val/test splits
        os.makedirs(dataset_dir / 'train', exist_ok=True)
        os.makedirs(dataset_dir / 'val', exist_ok=True)
        os.makedirs(dataset_dir / 'test', exist_ok=True)
        
        # Define some code templates for buggy and non-buggy samples
        buggy_templates = [
            # Off-by-one errors
            "def get_element(arr, index):\n    return arr[index]  # Missing bounds check",
            "def process_items(items):\n    for i in range(1, len(items)):  # Should start from 0\n        print(items[i])",
            "def calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)  # Division by zero if empty",
            # Null pointer equivalents
            "def process_data(data):\n    result = data['key']  # KeyError if key doesn't exist\n    return result * 2",
            # Uncaught exceptions
            "def parse_number(s):\n    return int(s)  # ValueError if s is not a number"
        ]
        
        non_buggy_templates = [
            "def get_element(arr, index):\n    if 0 <= index < len(arr):\n        return arr[index]\n    return None",
            "def process_items(items):\n    for i in range(len(items)):\n        print(items[i])",
            "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    total = sum(numbers)\n    return total / len(numbers)",
            "def process_data(data):\n    result = data.get('key', 0)\n    return result * 2",
            "def parse_number(s):\n    try:\n        return int(s)\n    except ValueError:\n        return None"
        ]
        
        # Generate synthetic data
        train_samples = self._generate_samples(buggy_templates, non_buggy_templates, 80)
        val_samples = self._generate_samples(buggy_templates, non_buggy_templates, 10)
        test_samples = self._generate_samples(buggy_templates, non_buggy_templates, 10)
        
        # Save to files
        with open(dataset_dir / 'train' / 'samples.json', 'w') as f:
            json.dump(train_samples, f, indent=2)
        
        with open(dataset_dir / 'val' / 'samples.json', 'w') as f:
            json.dump(val_samples, f, indent=2)
        
        with open(dataset_dir / 'test' / 'samples.json', 'w') as f:
            json.dump(test_samples, f, indent=2)
    
    def _generate_samples(self, buggy_templates, non_buggy_templates, count):
        """Generate synthetic code samples"""
        samples = []
        
        for i in range(count):
            if random.random() < 0.5:
                # Generate buggy sample
                code = random.choice(buggy_templates)
                # Add some variations to make samples diverse
                code = code.replace("arr", f"array_{i}")
                samples.append({
                    "id": f"sample_{i}",
                    "code": code,
                    "is_buggy": True
                })
            else:
                # Generate non-buggy sample
                code = random.choice(non_buggy_templates)
                # Add some variations
                code = code.replace("arr", f"array_{i}")
                samples.append({
                    "id": f"sample_{i}",
                    "code": code,
                    "is_buggy": False
                })
        
        return samples

    def _download_and_prepare_bugsinpy(self, dataset_dir: Path, max_bugs: Optional[int] = None) -> bool:
        """
        Download and prepare a BugsInPy-derived dataset. Attempts multiple
        BugsInPy APIs and extracts function-level before/after pairs when possible.

        Returns True if samples were written, False otherwise.
        """
        try:
            import bugsinpy
        except Exception as e:
            logger.warning("'bugsinpy' not installed or failed to import: %s", e)
            return False

        # Try to obtain an iterable of bug objects from the package (defensive)
        bugs = []
        try:
            if hasattr(bugsinpy, 'get_bugs'):
                bugs = bugsinpy.get_bugs()
            elif hasattr(bugsinpy, 'Bugs'):
                Bugs = bugsinpy.Bugs()
                if hasattr(Bugs, 'get_all'):
                    bugs = Bugs.get_all()
                elif hasattr(Bugs, 'list'):
                    bugs = Bugs.list()
            elif hasattr(bugsinpy, 'dataset'):
                try:
                    bugs = bugsinpy.dataset.get_bugs()
                except Exception:
                    bugs = []
        except Exception as e:
            logger.warning("Error enumerating BugsInPy bugs: %s", e)
            bugs = []

        if not bugs:
            logger.warning('No bugs discovered via BugsInPy API')
            return False

        samples = []

        def extract_functions_from_source(src: str):
            """Return list of (func_name, func_source) for top-level functions."""
            try:
                tree = ast.parse(src)
            except Exception:
                return []
            funcs = []
            lines = src.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Best-effort: use lineno/end_lineno to slice original source
                    try:
                        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                            start = node.lineno - 1
                            end = node.end_lineno
                            func_src = "\n".join(lines[start:end])
                        else:
                            func_src = ast.unparse(node) if hasattr(ast, 'unparse') else None
                    except Exception:
                        func_src = None
                    if func_src:
                        funcs.append((node.name, func_src))
            return funcs

        def add_pair(bug_id: str, src_before: str, src_after: str):
            before_funcs = extract_functions_from_source(src_before) if src_before else []
            after_funcs = extract_functions_from_source(src_after) if src_after else []

            before_map = {n: s for n, s in before_funcs}
            after_map = {n: s for n, s in after_funcs}

            common = set(before_map).intersection(set(after_map))
            if common:
                for func_name in common:
                    b_src = before_map[func_name]
                    a_src = after_map[func_name]
                    samples.append({'id': f"{bug_id}_{func_name}_buggy", 'code': b_src, 'is_buggy': True})
                    samples.append({'id': f"{bug_id}_{func_name}_fixed", 'code': a_src, 'is_buggy': False})
            else:
                # Fallback: include file-level before/after
                if src_before:
                    samples.append({'id': f"{bug_id}_file_buggy", 'code': src_before, 'is_buggy': True})
                if src_after:
                    samples.append({'id': f"{bug_id}_file_fixed", 'code': src_after, 'is_buggy': False})

        count = 0
        for b in bugs:
            if max_bugs and count >= max_bugs:
                break
            try:
                bug_id = getattr(b, 'bug_id', None) or getattr(b, 'id', None) or f"bug{count}"

                # Try common attributes/methods for before/after content
                src_before = None
                src_after = None
                for attr in ('before', 'buggy', 'get_before', 'get_buggy'):
                    try:
                        val = getattr(b, attr)
                        src_before = val() if callable(val) else val
                        if src_before:
                            break
                    except Exception:
                        src_before = None
                for attr in ('after', 'fixed', 'get_after', 'get_fixed'):
                    try:
                        val = getattr(b, attr)
                        src_after = val() if callable(val) else val
                        if src_after:
                            break
                    except Exception:
                        src_after = None

                # If both are None, try to inspect a patch/diff (best-effort)
                if not src_before and not src_after:
                    for meth in ('get_patch', 'get_diff', 'get_changes'):
                        if hasattr(b, meth):
                            try:
                                patch_text = getattr(b, meth)()
                                # We don't parse patches here automatically; skip
                            except Exception:
                                pass

                if not src_before and not src_after:
                    logger.debug(f"Skipping bug {bug_id}: no before/after source available")
                    count += 1
                    continue

                add_pair(bug_id, src_before or "", src_after or "")
                count += 1
            except Exception as e:
                logger.warning(f"Error processing bug object: {e}")
                continue

        if not samples:
            logger.warning('No samples extracted from BugsInPy')
            return False

        random.shuffle(samples)
        n = len(samples)
        t = int(0.8 * n)
        v = int(0.9 * n)

        with open(dataset_dir / 'train' / 'samples.json', 'w') as f:
            json.dump(samples[:t], f, indent=2)
        with open(dataset_dir / 'val' / 'samples.json', 'w') as f:
            json.dump(samples[t:v], f, indent=2)
        with open(dataset_dir / 'test' / 'samples.json', 'w') as f:
            json.dump(samples[v:], f, indent=2)

        logger.info(f"Saved BugsInPy-derived dataset to {dataset_dir} (samples={len(samples)})")
        return True
    
    def process_dataset(self, dataset_path: str) -> None:
        """
        Process the dataset to extract AST paths and build vocabularies
        
        Args:
            dataset_path: Path to the dataset directory
        """
        dataset_dir = Path(dataset_path)
        
        # Process the dataset
        logger.info(f"Processing dataset at {dataset_path}")
        
        # Read samples from JSON files
        train_samples = self._load_and_preprocess_samples(dataset_dir / 'train' / 'samples.json')
        val_samples = self._load_and_preprocess_samples(dataset_dir / 'val' / 'samples.json')
        test_samples = self._load_and_preprocess_samples(dataset_dir / 'test' / 'samples.json')
        
        # Build vocabulary from training samples
        path_contexts = []
        token_contexts = []
        for sample in train_samples:
            for pc in sample['path_contexts']:
                path_contexts.append(pc['path'])
                # collect tokens for token vocab
                if pc.get('start_token'):
                    token_contexts.append(pc['start_token'])
                if pc.get('end_token'):
                    token_contexts.append(pc['end_token'])
        
        # Build vocabularies
        unique_paths = sorted(set(path_contexts))
        self.path_vocab = {path: idx+1 for idx, path in enumerate(unique_paths)}  # 0 is reserved for padding

        # Build token vocabulary from terminal tokens in training data
        unique_tokens = sorted(set(token_contexts))
        self.token_vocab = {tok: idx+1 for idx, tok in enumerate(unique_tokens)}  # 0 reserved for padding
        
        # Save processed data
        processed_dir = dataset_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        with open(processed_dir / 'train_samples.pkl', 'wb') as f:
            pickle.dump(train_samples, f)
        
        with open(processed_dir / 'val_samples.pkl', 'wb') as f:
            pickle.dump(val_samples, f)
        
        with open(processed_dir / 'test_samples.pkl', 'wb') as f:
            pickle.dump(test_samples, f)
        
        with open(processed_dir / 'path_vocab.pkl', 'wb') as f:
            pickle.dump(self.path_vocab, f)
        with open(processed_dir / 'token_vocab.pkl', 'wb') as f:
            pickle.dump(self.token_vocab, f)
        
        logger.info(f"Processed dataset saved to {processed_dir}")
        logger.info(f"Path vocabulary size: {len(self.path_vocab)}")
    
    def _load_and_preprocess_samples(self, json_path: Path) -> List[Dict]:
        """
        Load samples from JSON and extract AST paths
        
        Args:
            json_path: Path to the JSON file with samples
            
        Returns:
            List of preprocessed samples with AST paths
        """
        with open(json_path, 'r') as f:
            samples = json.load(f)
        
        preprocessed_samples = []
        
        for sample in samples:
            try:
                code = sample['code']
                ast_tree = normalize_and_parse_code(code)
                path_contexts = extract_ast_paths(ast_tree, self.max_paths, self.max_path_length)
                
                preprocessed_samples.append({
                    'id': sample['id'],
                    'is_buggy': sample['is_buggy'],
                    'path_contexts': path_contexts
                })
            except SyntaxError:
                logger.warning(f"Syntax error in sample {sample['id']}, skipping")
            except Exception as e:
                logger.warning(f"Error processing sample {sample['id']}: {e}")
        
        return preprocessed_samples
    
    def load_datasets(self, dataset_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load processed datasets and create DataLoaders
        
        Args:
            dataset_path: Path to the processed dataset
            
        Returns:
            DataLoaders for train, validation and test sets
        """
        processed_dir = Path(dataset_path) / 'processed'
        
        if not processed_dir.exists():
            self.process_dataset(dataset_path)
        
        # Load processed samples
        with open(processed_dir / 'train_samples.pkl', 'rb') as f:
            train_samples = pickle.load(f)
        
        with open(processed_dir / 'val_samples.pkl', 'rb') as f:
            val_samples = pickle.load(f)
        
        with open(processed_dir / 'test_samples.pkl', 'rb') as f:
            test_samples = pickle.load(f)
        
        # Load vocabulary
        with open(processed_dir / 'path_vocab.pkl', 'rb') as f:
            self.path_vocab = pickle.load(f)
        # Load token vocabulary if present
        token_vocab_path = processed_dir / 'token_vocab.pkl'
        if token_vocab_path.exists():
            with open(token_vocab_path, 'rb') as f:
                self.token_vocab = pickle.load(f)
        
        # Create datasets
        self.train_dataset = CodeDataset(train_samples, self.path_vocab, self.token_vocab, self.max_paths)
        self.val_dataset = CodeDataset(val_samples, self.path_vocab, self.token_vocab, self.max_paths)
        self.test_dataset = CodeDataset(test_samples, self.path_vocab, self.token_vocab, self.max_paths)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=32)
        test_loader = DataLoader(self.test_dataset, batch_size=32)
        
        logger.info(f"Loaded datasets: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def get_vocab_size(self) -> int:
        """Return the size of the path vocabulary"""
        return len(self.path_vocab) + 1  # +1 for padding
