#!/usr/bin/env python3
"""
Script to compute and plot the cosine similarity between the original image's reversed PRC
and the blurred images' reversed PRCs.
"""

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def cosine_similarity(tensor1, tensor2):
    """
    Compute the cosine similarity between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        
    Returns:
        Cosine similarity value between 0 and 1
    """
    # Flatten tensors if they're not already 1D
    t1 = tensor1.flatten()
    t2 = tensor2.flatten()
    
    # Compute cosine similarity
    dot_product = torch.sum(t1 * t2)
    norm1 = torch.sqrt(torch.sum(t1 * t1))
    norm2 = torch.sqrt(torch.sum(t2 * t2))
    
    return dot_product / (norm1 * norm2)

def main():
    # Path to tensors directory
    tensors_dir = Path("tensors")
    
    # Load the original tensor
    original_tensor_path = tensors_dir / "1.pt"
    if not original_tensor_path.exists():
        print(f"Error: Original tensor file {original_tensor_path} not found")
        return
    
    original_tensor = torch.load(original_tensor_path)
    
    # Find all blurred tensor files
    blurred_files = []
    for file in os.listdir(tensors_dir):
        # Match files like "blurred_X.XX_1.pt" where X.XX is the sigma value
        match = re.match(r"blurred_(\d+\.\d+)_1\.pt", file)
        if match:
            sigma = float(match.group(1))
            blurred_files.append((sigma, tensors_dir / file))
    
    # Sort by sigma value
    blurred_files.sort(key=lambda x: x[0])
    
    if not blurred_files:
        print("No blurred tensor files found")
        return
    
    # Compute cosine similarity for each blurred tensor
    similarities = []
    for sigma, file_path in blurred_files:
        blurred_tensor = torch.load(file_path)
        similarity = cosine_similarity(original_tensor, blurred_tensor)
        similarities.append((sigma, similarity.item()))
        print(f"Sigma = {sigma}, Cosine Similarity = {similarity.item():.4f}")
    
    # Extract sigma values and similarity scores for plotting
    sigma_values = [s[0] for s in similarities]
    similarity_scores = [s[1] for s in similarities]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, similarity_scores, 'o-', linewidth=2, markersize=8)
    
    # Add a vertical line at the threshold sigma value
    threshold_sigma = 4.02
    plt.axvline(x=threshold_sigma, color='r', linestyle='--', label=f'Threshold σ = {threshold_sigma}')
    
    # Add labels and title
    plt.xlabel('Sigma (σ) Value', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Cosine Similarity vs. Blur Intensity (σ)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ensure y-axis starts from a reasonable value
    y_min = max(0, min(similarity_scores) - 0.05)
    y_max = min(1, max(similarity_scores) + 0.05)
    plt.ylim(y_min, y_max)
    
    # Add annotations for key points
    for i, (sigma, similarity) in enumerate(similarities):
        if i % 2 == 0 or sigma == threshold_sigma or i == len(similarities) - 1:
            plt.annotate(f'σ={sigma}\n{similarity:.4f}', 
                         xy=(sigma, similarity),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=9)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('cosine_similarity_plot.png', dpi=300)
    print("Plot saved as 'cosine_similarity_plot.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 