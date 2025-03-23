#!/usr/bin/env python3
"""
Script to run adversarial perturbation search on multiple watermarked images in batch mode.
"""

import os
import argparse
import torch
import pickle
import glob
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import subprocess
import time

# Set up argument parser
parser = argparse.ArgumentParser('Batch Adversarial Perturbation Search')
parser.add_argument('--images_dir', type=str, required=True, 
                    help='Directory containing watermarked images')
parser.add_argument('--output_dir', type=str, default='adversarial_results', 
                    help='Directory to save results')
parser.add_argument('--exp_id', type=str, default='prc_num_10_steps_50_fpr_1e-05_nowm_0', 
                    help='Experiment ID for loading keys')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='Learning rate for optimization')
parser.add_argument('--iterations', type=int, default=1000, 
                    help='Maximum number of iterations')
parser.add_argument('--l2_weight', type=float, default=0.1, 
                    help='Weight for L2 norm penalty')
parser.add_argument('--ssim_weight', type=float, default=0.5, 
                    help='Weight for SSIM penalty')
parser.add_argument('--detection_threshold', type=float, default=0.5, 
                    help='Cosine similarity threshold for detection')
parser.add_argument('--max_workers', type=int, default=1, 
                    help='Maximum number of parallel workers (use 1 for GPU memory constraints)')
parser.add_argument('--file_pattern', type=str, default='*.png', 
                    help='Pattern to match image files')
parser.add_argument('--analyze', action='store_true', 
                    help='Run analysis after completion')
args = parser.parse_args()

def run_adversarial_search(image_path):
    """Run adversarial perturbation search on a single image"""
    cmd = [
        'python', 'adversarial_perturbation.py',
        '-f', image_path,
        '--output_dir', args.output_dir,
        '--lr', str(args.lr),
        '--iterations', str(args.iterations),
        '--l2_weight', str(args.l2_weight),
        '--ssim_weight', str(args.ssim_weight),
        '--detection_threshold', str(args.detection_threshold),
        '--exp_id', args.exp_id
    ]
    
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error processing {image_path}:")
        print(process.stderr)
        return False
    
    print(f"Completed processing {image_path}")
    return True

def main():
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all image files
    image_files = glob.glob(os.path.join(args.images_dir, args.file_pattern))
    
    if not image_files:
        print(f"No images found in {args.images_dir} matching pattern {args.file_pattern}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    if args.max_workers > 1:
        print(f"Processing images in parallel with {args.max_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(tqdm(executor.map(run_adversarial_search, image_files), total=len(image_files)))
    else:
        print("Processing images sequentially")
        results = []
        for image_file in tqdm(image_files):
            results.append(run_adversarial_search(image_file))
    
    # Print summary
    success_count = sum(1 for r in results if r)
    print(f"Processed {len(image_files)} images, {success_count} successful, {len(image_files) - success_count} failed")
    
    # Run analysis if requested
    if args.analyze:
        print("Running analysis...")
        analysis_cmd = [
            'python', 'analyze_perturbations.py',
            '--results_dir', args.output_dir,
            '--output_dir', f"{args.output_dir}_analysis"
        ]
        subprocess.run(analysis_cmd)
        print(f"Analysis complete. Results saved to {args.output_dir}_analysis")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds") 