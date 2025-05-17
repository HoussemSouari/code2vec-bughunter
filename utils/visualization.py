"""
Visualization utilities for model attention and code analysis.
"""

import os
import logging
from typing import List, Tuple, Dict, Any
import html
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def visualize_attention(code: str, 
                       attention: List[Tuple[str, float]], 
                       output_path: str) -> None:
    """
    Create HTML visualization of code with attention highlights
    
    Args:
        code: Source code string
        attention: List of (ast_path, weight) tuples
        output_path: Path to save the HTML visualization
    """
    # Create HTML output
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Attention Visualization</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            pre {{
                margin: 0;
                padding: 0;
            }}
            .code-line {{
                padding: 2px 5px;
                margin: 0;
                white-space: pre;
                tab-size: 4;
            }}
            .attention-high {{
                background-color: rgba(255, 0, 0, 0.3);
            }}
            .attention-medium {{
                background-color: rgba(255, 165, 0, 0.3);
            }}
            .attention-low {{
                background-color: rgba(255, 255, 0, 0.2);
            }}
            .code-container {{
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .line-number {{
                user-select: none;
                color: #999;
                text-align: right;
                padding-right: 10px;
                padding-left: 5px;
                border-right: 1px solid #ddd;
                background-color: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1>Code Attention Visualization</h1>
            <p>This visualization shows the model's attention to different parts of the code when making a prediction.</p>
            
            <div class="row">
                <div class="col-md-8">
                    <h3>Code with Attention Highlights</h3>
                    <div class="code-container">
                        <table class="table table-sm mb-0">
                            <tbody>
    """
    
    # Add code with line numbers
    lines = code.split('\n')
    for i, line in enumerate(lines):
        escaped_line = html.escape(line)
        html_content += f"""
                                <tr>
                                    <td class="line-number">{i+1}</td>
                                    <td class="code-line" id="line-{i+1}">{escaped_line}</td>
                                </tr>
        """
    
    html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <h3>Top Attention Areas</h3>
                    <div class="list-group">
    """
    
    # Add attention items
    for i, (path, weight) in enumerate(attention[:10], 1):
        html_content += f"""
                        <div class="list-group-item">
                            <div class="d-flex w-100 justify-content-between">
                                <h5 class="mb-1">Path {i}</h5>
                                <small>{weight:.4f}</small>
                            </div>
                            <p class="mb-1">{html.escape(path)}</p>
                        </div>
        """
    
    html_content += """
                    </div>
                </div>
            </div>
            
            <h3 class="mt-4">Attention Distribution</h3>
            <div id="attention-chart" style="height: 300px;"></div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            // Create attention chart
            const attentionData = [
    """
    
    # Add chart data
    for i, (path, weight) in enumerate(attention[:15]):
        label = f"Path {i+1}"
        html_content += f"                {{label: '{label}', value: {weight:.4f}}},\n"
    
    html_content += """
            ];
            
            // Initialize chart
            window.onload = function() {
                const ctx = document.createElement('canvas');
                document.getElementById('attention-chart').appendChild(ctx);
                
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: attentionData.map(d => d.label),
                        datasets: [{
                            label: 'Attention Weight',
                            data: attentionData.map(d => d.value),
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            };
        </script>
    </body>
    </html>
    """
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Visualization saved to {output_path}")


def plot_attention_heatmap(code: str, 
                          attention_weights: np.ndarray, 
                          output_path: str) -> None:
    """
    Plot attention heatmap for visualization
    
    Args:
        code: Source code string
        attention_weights: Attention weights for code elements
        output_path: Path to save the plot
    """
    # Split code into lines
    lines = code.split('\n')
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, len(lines) * 0.3))
    
    # Plot heatmap
    im = ax.imshow(attention_weights.reshape(-1, 1), cmap='YlOrRd', aspect='auto')
    
    # Set x and y ticks
    ax.set_yticks(range(len(lines)))
    ax.set_yticklabels(lines)
    ax.set_xticks([])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set title
    plt.title('Attention Weights for Code')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Attention heatmap saved to {output_path}")
