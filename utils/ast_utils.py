

import ast
import networkx as nx
from typing import List, Dict, Tuple, Set, Generator, Any
import re


def normalize_code(code: str) -> str:
    """
    Normalize code by removing comments and normalizing whitespace
    
    Args:
        code: Source code string
        
    Returns:
        Normalized code string
    """
    # Remove comments
    lines = code.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Remove comments
        if '#' in line:
            line = line[:line.index('#')]
        normalized_lines.append(line.rstrip())
    
    return '\n'.join(normalized_lines)


def normalize_and_parse_code(code: str) -> ast.AST:
    """
    Normalize and parse code into an AST
    
    Args:
        code: Source code string
        
    Returns:
        AST node
    """
    # Normalize code
    normalized_code = normalize_code(code)
    
    # Parse AST
    return ast.parse(normalized_code)


def build_ast_graph(tree: ast.AST) -> nx.DiGraph:
    """
    Build a graph representation of the AST
    
    Args:
        tree: AST node
        
    Returns:
        NetworkX directed graph
    """
    graph = nx.DiGraph()
    
    # Helper function to add nodes and edges
    def add_nodes_edges(node, parent=None):
        node_id = id(node)
        node_type = type(node).__name__
        # Derive a human-readable label for terminal nodes
        try:
            node_label = get_node_label(node)
        except Exception:
            node_label = None

        # Add node to graph with type and optional label/value/lineno
        attrs = {'type': node_type}
        if node_label is not None:
            attrs['label'] = node_label
        if hasattr(node, 'lineno'):
            attrs['lineno'] = node.lineno
        # For Constant/Num/Str, store the raw value when available
        if isinstance(node, ast.Constant):
            attrs['value'] = getattr(node, 'value', None)
        elif isinstance(node, ast.Num):
            attrs['value'] = getattr(node, 'n', None)
        elif isinstance(node, ast.Str):
            attrs['value'] = getattr(node, 's', None)

        graph.add_node(node_id, **attrs)
        
        # Add edge from parent
        if parent is not None:
            graph.add_edge(parent, node_id)
        
        # Visit children
        for child_name, child in ast.iter_fields(node):
            if isinstance(child, ast.AST):
                add_nodes_edges(child, node_id)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ast.AST):
                        add_nodes_edges(item, node_id)
    
    # Build graph
    add_nodes_edges(tree)
    
    return graph


def get_node_label(node: ast.AST) -> str:
    """
    Get a label for an AST node
    
    Args:
        node: AST node
        
    Returns:
        String label for the node
    """
    node_type = type(node).__name__
    
    # For terminal nodes, include the value
    if isinstance(node, ast.Name):
        return f"Name:{node.id}"
    elif isinstance(node, ast.Num):
        return f"Num:{node.n}"
    elif isinstance(node, ast.Str):
        # Truncate long strings
        s = str(node.s)
        if len(s) > 10:
            s = s[:10] + "..."
        return f"Str:{s}"
    elif isinstance(node, ast.Constant):
        # For Python 3.8+
        if isinstance(node.value, str):
            s = node.value
            if len(s) > 10:
                s = s[:10] + "..."
            return f"Str:{s}"
        return f"Constant:{node.value}"
    
    # For non-terminal nodes, just use the type
    return node_type


def extract_paths(graph: nx.DiGraph, 
                 max_paths: int = 100, 
                 max_length: int = 8) -> List[Dict[str, Any]]:
    """
    Extract path contexts from AST graph
    
    Args:
        graph: NetworkX directed graph of AST
        max_paths: Maximum number of paths to extract
        max_length: Maximum length of paths
        
    Returns:
        List of path contexts (start token, path, end token)
    """
    paths = []

    # Get terminal nodes (leaf nodes)
    terminal_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]

    # Helper: get a label for a node id using attributes stored in the graph
    def node_label_from_graph(node_id):
        node_data = graph.nodes.get(node_id, {})
        # If a precomputed label exists, use it
        if 'label' in node_data:
            return node_data['label']
        # Otherwise, attempt to derive a label from the type and any stored value
        node_type = node_data.get('type')
        if node_type is None:
            return 'Unknown'
        # For leaf-like nodes, a 'value' attribute may be present
        value = node_data.get('value')
        if value is not None:
            return f"{node_type}:{value}"
        return node_type

    # For each pair of terminal nodes, find the shortest path
    for i, source in enumerate(terminal_nodes):
        for target in terminal_nodes[i+1:]:
            try:
                # Find shortest path between terminals in undirected AST
                path = nx.shortest_path(graph.to_undirected(), source, target)

                # Skip if path is too long
                if len(path) > max_length:
                    continue

                # Get path as node types
                path_types = [graph.nodes[n].get('type', 'Unknown') for n in path]

                # Create path string (e.g., Name->Assign->Num)
                path_str = '->'.join(path_types)

                # Get terminal node labels using stored attributes
                source_label = node_label_from_graph(source)
                target_label = node_label_from_graph(target)

                # Add to paths
                paths.append({
                    'start_token': source_label,
                    'path': path_str,
                    'end_token': target_label
                })

                # Stop if we have enough paths
                if len(paths) >= max_paths:
                    return paths

            except nx.NetworkXNoPath:
                # No path between nodes
                continue

    return paths


def extract_ast_paths(tree: ast.AST, 
                     max_paths: int = 100, 
                     max_length: int = 8) -> List[Dict[str, Any]]:
    """
    Extract path contexts from AST
    
    Args:
        tree: AST node
        max_paths: Maximum number of paths to extract
        max_length: Maximum length of paths
        
    Returns:
        List of path contexts (start token, path, end token)
    """
    # Build AST graph
    graph = build_ast_graph(tree)
    
    # Extract paths
    return extract_paths(graph, max_paths, max_length)
