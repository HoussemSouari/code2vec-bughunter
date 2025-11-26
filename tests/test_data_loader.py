

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path

import torch
from scripts.data_loader import DataManager, CodeDataset


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading and processing"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(data_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_download_dataset(self):
        """Test dataset download/creation"""
        dataset_path = self.data_manager.download_dataset('defects4j_subset')
        
        # Check that dataset exists
        self.assertTrue(os.path.exists(dataset_path))
        
        # Check that train/val/test directories are created
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'train')))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'val')))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'test')))
        
        # Check that sample files exist
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'train', 'samples.json')))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'val', 'samples.json')))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, 'test', 'samples.json')))
    
    def test_process_dataset(self):
        """Test dataset processing"""
        # Download dataset
        dataset_path = self.data_manager.download_dataset('defects4j_subset')
        
        # Process dataset
        self.data_manager.process_dataset(dataset_path)
        
        # Check that processed directory and files exist
        processed_dir = os.path.join(dataset_path, 'processed')
        self.assertTrue(os.path.exists(processed_dir))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, 'train_samples.pkl')))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, 'val_samples.pkl')))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, 'test_samples.pkl')))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, 'path_vocab.pkl')))
    
    def test_load_datasets(self):
        """Test loading datasets into DataLoaders"""
        # Download and process dataset
        dataset_path = self.data_manager.download_dataset('defects4j_subset')
        
        # Load datasets
        train_loader, val_loader, test_loader = self.data_manager.load_datasets(dataset_path)
        
        # Check that DataLoaders are created properly
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Check that data can be retrieved from DataLoader
        batch = next(iter(train_loader))
        self.assertIn('paths', batch)
        self.assertIn('label', batch)
        
    def test_code_dataset(self):
        """Test CodeDataset class"""
        # Create simple path vocabulary
        path_vocab = {'path1': 1, 'path2': 2, 'path3': 3}
        
        # Create samples
        samples = [
            {
                'id': 'sample1',
                'is_buggy': True,
                'path_contexts': [
                    {'start_token': 'a', 'path': 'path1', 'end_token': 'b'},
                    {'start_token': 'c', 'path': 'path2', 'end_token': 'd'}
                ]
            },
            {
                'id': 'sample2',
                'is_buggy': False,
                'path_contexts': [
                    {'start_token': 'e', 'path': 'path3', 'end_token': 'f'},
                    {'start_token': 'g', 'path': 'path1', 'end_token': 'h'}
                ]
            }
        ]
        
        # Create dataset
        dataset = CodeDataset(samples, path_vocab, max_paths=5)
        
        # Check dataset size
        self.assertEqual(len(dataset), 2)
        
        # Get item from dataset
        item = dataset[0]
        
        # Check item format
        self.assertIn('paths', item)
        self.assertIn('label', item)
        
        # Check shapes
        self.assertEqual(item['paths'].shape, (5,))  # max_paths = 5
        self.assertEqual(item['label'].shape, ())  # scalar
        
        # Check values
        self.assertEqual(item['paths'][0].item(), 1)  # path1 has index 1
        self.assertEqual(item['paths'][1].item(), 2)  # path2 has index 2
        self.assertEqual(item['label'].item(), 1.0)  # is_buggy = True


if __name__ == '__main__':
    unittest.main()
