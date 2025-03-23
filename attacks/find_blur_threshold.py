#!/usr/bin/env python3
"""
Script to find the threshold sigma value for Gaussian blur that allows watermark detection to still work.
This script applies Gaussian blur with increasing sigma values and checks if the watermark can still be detected.
"""

import argparse
import os
import subprocess
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

# Define our own gaussian_blur function instead of importing from transform.py
# This avoids the argument parser conflict
def gaussian_blur(img, sigma=2.0):
    """
    Apply Gaussian blur to an image while preserving its overall appearance.
    
    Args:
        img: PIL Image object
        sigma: Blur radius/sigma value (higher = more blur)
        
    Returns:
        PIL Image with Gaussian blur applied
    """
    # Create a copy of the image to avoid modifying the original
    blurred_img = img.copy()
    
    # Apply Gaussian blur with the specified radius
    blurred_img = blurred_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    return blurred_img

def parse_args():
    parser = argparse.ArgumentParser(description='Find threshold sigma value for Gaussian blur')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the watermarked image')
    parser.add_argument('--min_sigma', type=float, default=1.0, help='Minimum sigma value to test')
    parser.add_argument('--max_sigma', type=float, default=10.0, help='Maximum sigma value to test')
    parser.add_argument('--step', type=float, default=0.5, help='Step size for sigma values')
    parser.add_argument('--binary_search', action='store_true', help='Use binary search instead of linear search')
    return parser.parse_args()

def check_detection(image_path):
    """Run decode.py with the given image and check if watermark is detected"""
    result = subprocess.run(['python', 'decode.py', '-f', image_path], 
                           capture_output=True, text=True)
    
    # Parse the output to determine if detection was successful
    output = result.stdout
    print(f"decode.py output: {output}")
    
    # Look for detection result in the output
    detection_result = False
    combined_result = False
    
    for line in output.split('\n'):
        if 'Detection:' in line and 'Combined:' in line:
            # Parse line like: "0: Detection: True; Decoding: False; Combined: True"
            if 'Detection: True' in line:
                detection_result = True
            if 'Combined: True' in line:
                combined_result = True
            break
    
    # Return True if either detection or combined result is True
    return detection_result or combined_result

def linear_search(image_path, min_sigma, max_sigma, step):
    """
    Perform a linear search to find the threshold sigma value.
    Tests sigma values from min_sigma to max_sigma with the given step size.
    """
    print(f"Performing linear search from sigma={min_sigma} to {max_sigma} with step={step}")
    
    original_image = Image.open(image_path)
    image_dir = Path(image_path).parent
    image_name = Path(image_path).name
    
    results = []
    
    for sigma in np.arange(min_sigma, max_sigma + step, step):
        sigma = round(sigma, 2)  # Round to 2 decimal places for cleaner output
        print(f"\nTesting sigma = {sigma}")
        
        # Apply Gaussian blur
        blurred_image = gaussian_blur(original_image, sigma)
        
        # Save the blurred image
        blurred_path = f"{image_dir}/blurred_{sigma}_{image_name}"
        blurred_image.save(blurred_path)
        print(f"Saved blurred image to {blurred_path}")
        
        # Check if watermark is still detectable
        detected = check_detection(blurred_path)
        results.append((sigma, detected))
        
        print(f"Sigma = {sigma}, Detection = {detected}")
        
        # If this is the first failure, we've found our approximate threshold
        if not detected and all(result[1] for result in results[:-1]):
            print(f"\nFound threshold: sigma = {results[-2][0]} (detected) -> {sigma} (not detected)")
            break
    
    # Print summary of results
    print("\nResults summary:")
    for sigma, detected in results:
        print(f"Sigma = {sigma}, Detection = {detected}")
    
    # Determine the threshold
    if all(result[1] for result in results):
        print(f"\nWatermark was detected for all tested sigma values up to {max_sigma}")
        return max_sigma
    elif not any(result[1] for result in results):
        print(f"\nWatermark was not detected for any tested sigma values, even at {min_sigma}")
        return min_sigma
    else:
        # Find the threshold (last successful detection)
        for i in range(len(results) - 1):
            if results[i][1] and not results[i+1][1]:
                threshold = results[i][0]
                print(f"\nThreshold sigma value: {threshold}")
                return threshold

def binary_search(image_path, min_sigma, max_sigma, precision=0.1):
    """
    Perform a binary search to find the threshold sigma value.
    More efficient than linear search for finding the exact threshold.
    """
    print(f"Performing binary search between sigma={min_sigma} and {max_sigma}")
    
    original_image = Image.open(image_path)
    image_dir = Path(image_path).parent
    image_name = Path(image_path).name
    
    low = min_sigma
    high = max_sigma
    threshold = None
    
    # First check if detection works at min_sigma
    blurred_image = gaussian_blur(original_image, low)
    blurred_path = f"{image_dir}/blurred_{low}_{image_name}"
    blurred_image.save(blurred_path)
    if not check_detection(blurred_path):
        print(f"Watermark not detected even at minimum sigma={low}")
        return low
    
    # Then check if detection fails at max_sigma
    blurred_image = gaussian_blur(original_image, high)
    blurred_path = f"{image_dir}/blurred_{high}_{image_name}"
    blurred_image.save(blurred_path)
    if check_detection(blurred_path):
        print(f"Watermark still detected at maximum sigma={high}")
        return high
    
    # Binary search for the threshold
    while high - low > precision:
        mid = (low + high) / 2
        mid = round(mid, 2)  # Round to 2 decimal places
        
        print(f"\nTesting sigma = {mid}")
        
        # Apply Gaussian blur
        blurred_image = gaussian_blur(original_image, mid)
        
        # Save the blurred image
        blurred_path = f"{image_dir}/blurred_{mid}_{image_name}"
        blurred_image.save(blurred_path)
        
        # Check if watermark is still detectable
        detected = check_detection(blurred_path)
        print(f"Sigma = {mid}, Detection = {detected}")
        
        if detected:
            low = mid  # If detected, threshold is higher
            threshold = mid  # Update the last known working threshold
        else:
            high = mid  # If not detected, threshold is lower
    
    print(f"\nThreshold sigma value: {threshold}")
    return threshold

def main():
    args = parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    print(f"Finding blur threshold for image: {args.file}")
    
    if args.binary_search:
        threshold = binary_search(args.file, args.min_sigma, args.max_sigma)
    else:
        threshold = linear_search(args.file, args.min_sigma, args.max_sigma, args.step)
    
    print(f"\nFinal threshold sigma value: {threshold}")
    print(f"Watermark detection works up to sigma={threshold}")

if __name__ == "__main__":
    main() 