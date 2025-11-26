
import unittest
import ast
from utils.ast_utils import normalize_code, normalize_and_parse_code, extract_ast_paths


class TestASTUtils(unittest.TestCase):
    """Test cases for AST utilities"""
    
    def test_normalize_code(self):
        """Test code normalization"""
        code = "x = 1  # This is a comment\ny = 2\n"
        normalized = normalize_code(code)
        self.assertEqual(normalized, "x = 1\ny = 2")
    
    def test_normalize_and_parse_code(self):
        """Test code normalization and parsing"""
        code = "x = 1\ny = 2\n"
        tree = normalize_and_parse_code(code)
        self.assertIsInstance(tree, ast.AST)
        self.assertEqual(len(tree.body), 2)
    
    def test_extract_ast_paths(self):
        """Test AST path extraction"""
        code = """
def example(x):
    if x > 0:
        return x * 2
    else:
        return x
"""
        tree = normalize_and_parse_code(code)
        paths = extract_ast_paths(tree, max_paths=10, max_length=8)
        
        # Verify that we have paths
        self.assertTrue(len(paths) > 0)
        
        # Verify path format
        for path in paths:
            self.assertIn('start_token', path)
            self.assertIn('path', path)
            self.assertIn('end_token', path)
    
    def test_extract_ast_paths_complex(self):
        """Test AST path extraction on more complex code"""
        code = """
def complex_function(a, b, c=None):
    result = 0
    if a > b:
        result = a * b
    elif a == b:
        result = a + b
    else:
        result = b - a
    
    if c is not None:
        result += c
    
    return result
"""
        tree = normalize_and_parse_code(code)
        paths = extract_ast_paths(tree, max_paths=20, max_length=8)
        
        # Verify that we have paths
        self.assertTrue(len(paths) > 0)
        
        # Check if we have appropriate path representation
        path_strs = [p['path'] for p in paths]
        # We should find paths containing If nodes
        self.assertTrue(any('If' in p for p in path_strs))
    
    def test_extract_ast_paths_error_handling(self):
        """Test AST path extraction error handling"""
        # Test with invalid code
        invalid_code = "def invalid_function(x:\n    return x"
        
        # This should raise a SyntaxError
        with self.assertRaises(SyntaxError):
            tree = normalize_and_parse_code(invalid_code)
            paths = extract_ast_paths(tree, max_paths=10, max_length=8)


if __name__ == '__main__':
    unittest.main()
