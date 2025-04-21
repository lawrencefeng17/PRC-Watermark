"""
Watermark Robustness Test Against Crop-Resize Attacks

This script tests the robustness of watermarking detection against crop-and-resize attacks.
It progressively reduces the crop size to find the threshold at which watermarks fail.
"""

import argparse
import os
import pickle
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add the project root directory to Python path so 'src' can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to the project root
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to Python path")

from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, exact_inversion

# Include the crop_and_resize function from the original script
def crop_and_resize(img, section_width, section_height):
    """
    Crop the top-left part of the image and resize it to the original dimensions.
    
    Args:
        img: PIL Image object
        section_width: Width of the section to crop
        section_height: Height of the section to crop
        
    Returns:
        PIL Image with the top-left section cropped and resized to original dimensions
    """
    # Get image dimensions
    img_width, img_height = img.size
    
    # Ensure section dimensions don't exceed image dimensions
    section_width = min(section_width, img_width)
    section_height = min(section_height, img_height)
    
    # Crop the top-left section
    section = img.crop((0, 0, section_width, section_height))
    
    # Resize the cropped section to the original image dimensions
    resized_section = section.resize((img_width, img_height), Image.LANCZOS)
    
    return resized_section

parser = argparse.ArgumentParser('Watermark Robustness Test Against Crop-Resize Attacks')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing watermarked input images')
parser.add_argument('--exp_id', type=str, required=True, 
                    help='Experiment ID for loading the watermark keys')
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base', 
                    help='Model ID used for watermarking and inversion')
parser.add_argument('--inf_steps', type=int, default=50, help='Number of inference steps')
parser.add_argument('--output_dir', type=str, default='crop_resize_test_results', 
                    help='Output directory for test results')
parser.add_argument('--var', type=float, default=1.5, help='Variance for PRC watermark detection')
parser.add_argument('--num_images', type=int, default=10, help='Number of images to test')
parser.add_argument('--min_crop_percent', type=int, default=10, 
                    help='Minimum crop size as percentage of original dimensions')
parser.add_argument('--max_crop_percent', type=int, default=90, 
                    help='Maximum crop size as percentage of original dimensions')
parser.add_argument('--step_size', type=int, default=10, 
                    help='Step size for crop percentage increments')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed for reproducibility
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(args.seed)

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_cache_dir = os.path.expanduser('~/.cache/huggingface/hub')

# Load watermark keys
with open(f'keys/{args.exp_id}.pkl', 'rb') as f:
    encoding_key, decoding_key = pickle.load(f)
    print(f'Loaded PRC keys from file keys/{args.exp_id}.pkl')

# Load the model for inversion
pipe = stable_diffusion_pipe(solver_order=1, model_id=args.model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

# Get list of image files in the input directory
image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
if args.num_images < len(image_files):
    image_files = image_files[:args.num_images]  # Take the first num_images

# Create a results dictionary
results = {
    "model": args.model_id,
    "exp_id": args.exp_id,
    "crop_percentages": list(range(args.max_crop_percent, args.min_crop_percent - 1, -args.step_size)),
    "image_results": {},
    "summary": {}
}

# Prepare directories for transformed images
for percent in results["crop_percentages"]:
    crop_dir = os.path.join(args.output_dir, f"crop_{percent}percent")
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

# First, verify that the original images have watermarks
print("\nVerifying watermarks in original images...")
original_watermarks = {}

for image_file in tqdm(image_files, desc="Checking original images"):
    image_path = os.path.join(args.input_dir, image_file)
    input_img = Image.open(image_path)
    input_filename = Path(image_file).stem
    
    # Get the reversed latents for the original image
    original_reversed_latents = exact_inversion(
        input_img,
        prompt='',
        test_num_inference_steps=args.inf_steps,
        inv_order=0,
        pipe=pipe
    )
    
    # Recover posteriors for watermark detection
    original_reversed_prc = prc_gaussians.recover_posteriors(
        original_reversed_latents.to(torch.float64).flatten().cpu(), 
        variances=float(args.var)
    ).flatten().cpu()
    
    # Check for watermark
    original_detection = Detect(decoding_key, original_reversed_prc)
    original_decoding = (Decode(decoding_key, original_reversed_prc) is not None)
    original_watermark = original_detection or original_decoding
    
    original_watermarks[image_file] = bool(original_watermark)
    
    if not original_watermark:
        print(f"Warning: No watermark detected in original image {image_file}")

# Filter to only use images that have watermarks
watermarked_images = [img for img in image_files if original_watermarks[img]]
print(f"Found {len(watermarked_images)} images with watermarks out of {len(image_files)} total.")

if not watermarked_images:
    print("Error: No watermarked images found. Please check your input directory and watermark keys.")
    exit(1)

# Process each image for each crop percentage
for image_file in tqdm(watermarked_images, desc="Processing images"):
    image_path = os.path.join(args.input_dir, image_file)
    input_img = Image.open(image_path)
    input_filename = Path(image_file).stem
    img_width, img_height = input_img.size
    
    results["image_results"][image_file] = {
        "original_size": [img_width, img_height],
        "crop_results": {}
    }
    
    # Test different crop percentages
    for crop_percent in tqdm(results["crop_percentages"], desc=f"Testing {image_file}", leave=False):
        # Calculate crop dimensions
        crop_width = int(img_width * crop_percent / 100)
        crop_height = int(img_height * crop_percent / 100)
        
        # Apply crop and resize
        transformed_img = crop_and_resize(input_img, crop_width, crop_height)
        
        # Save the transformed image
        output_path = os.path.join(args.output_dir, f"crop_{crop_percent}percent", f"{input_filename}_cropped.png")
        transformed_img.save(output_path)
        
        # Run exact inversion on the transformed image
        transformed_latents = exact_inversion(
            transformed_img,
            prompt='',
            test_num_inference_steps=args.inf_steps,
            inv_order=0,
            pipe=pipe
        )
        
        # Recover posteriors for watermark detection
        transformed_prc = prc_gaussians.recover_posteriors(
            transformed_latents.to(torch.float64).flatten().cpu(), 
            variances=float(args.var)
        ).flatten().cpu()
        
        # Check for watermark in transformed image
        detection_result = Detect(decoding_key, transformed_prc)
        decoding_result = (Decode(decoding_key, transformed_prc) is not None)
        watermark_detected = detection_result or decoding_result
        
        # Save the PRC for later analysis
        prc_path = os.path.join(args.output_dir, f"crop_{crop_percent}percent", f"{input_filename}_prc.pt")
        torch.save(transformed_prc, prc_path)
        
        # Record results
        results["image_results"][image_file]["crop_results"][str(crop_percent)] = {
            "crop_dimensions": [crop_width, crop_height],
            "transformed_image": output_path,
            "watermark_detected": bool(watermark_detected),
            "detection_result": bool(detection_result),
            "decoding_result": bool(decoding_result)
        }
        
        # Print result for this test
        status = "Detected" if watermark_detected else "Not detected"
        print(f"  Crop {crop_percent}%: {status}")

# Calculate summary statistics
for crop_percent in results["crop_percentages"]:
    successful_detections = sum(
        1 for img in results["image_results"].values() 
        if results["image_results"][img]["crop_results"][str(crop_percent)]["watermark_detected"]
    )
    total_images = len(results["image_results"])
    detection_rate = successful_detections / total_images if total_images > 0 else 0
    
    results["summary"][str(crop_percent)] = {
        "detection_rate": detection_rate,
        "successful_detections": successful_detections,
        "total_images": total_images
    }

# Save results to JSON file
results_path = os.path.join(args.output_dir, "crop_resize_test_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

# Generate visualization of results
percentages = sorted([int(p) for p in results["summary"].keys()])
detection_rates = [results["summary"][str(p)]["detection_rate"] for p in percentages]

plt.figure(figsize=(10, 6))
plt.plot(percentages, detection_rates, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Crop Size (% of original dimensions)')
plt.ylabel('Watermark Detection Rate')
plt.title('Watermark Robustness Against Crop-Resize Attacks')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.05)

# Add horizontal line at 0.5 detection rate to help identify threshold
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.text(percentages[0], 0.52, '50% Detection Rate', color='r')

# Find the threshold where detection rate crosses 0.5
threshold_found = False
threshold_value = None

for i in range(1, len(percentages)):
    if (detection_rates[i-1] >= 0.5 and detection_rates[i] < 0.5) or (detection_rates[i-1] < 0.5 and detection_rates[i] >= 0.5):
        # Linear interpolation to find more precise threshold
        x1, y1 = percentages[i-1], detection_rates[i-1]
        x2, y2 = percentages[i], detection_rates[i]
        if x1 != x2:  # Avoid division by zero
            threshold_value = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
            threshold_found = True
            break

if threshold_found and threshold_value is not None:
    plt.axvline(x=threshold_value, color='g', linestyle='--', alpha=0.7)
    plt.text(threshold_value + 1, 0.3, f'Threshold: {threshold_value:.1f}%', color='g', rotation=90)

plt.savefig(os.path.join(args.output_dir, "detection_rate_vs_crop_size.png"), dpi=300, bbox_inches='tight')

# Generate a report with findings
report_path = os.path.join(args.output_dir, "crop_resize_test_report.txt")
with open(report_path, "w") as f:
    f.write("Watermark Robustness Against Crop-Resize Attacks\n")
    f.write("==============================================\n\n")
    
    f.write(f"Model: {args.model_id}\n")
    f.write(f"Experiment ID: {args.exp_id}\n")
    f.write(f"Number of images tested: {len(results['image_results'])}\n\n")
    
    f.write("Summary of Results:\n")
    f.write("-----------------\n")
    f.write(f"{'Crop Size (%)':<15} | {'Detection Rate':<15} | {'Successful/Total'}\n")
    f.write("-" * 60 + "\n")
    
    for percent in sorted([int(p) for p in results["summary"].keys()]):
        summary = results["summary"][str(percent)]
        f.write(f"{percent:<15} | {summary['detection_rate']:<15.2%} | {summary['successful_detections']}/{summary['total_images']}\n")
    
    f.write("\nFindings:\n")
    f.write("---------\n")
    
    if threshold_found and threshold_value is not None:
        f.write(f"The watermark detection threshold appears to be around {threshold_value:.1f}% crop size.\n")
        f.write(f"This means that when less than {threshold_value:.1f}% of the original image is used in the crop-resize attack, the watermark detection rate falls below 50%.\n\n")
    else:
        # Check if always above or always below threshold
        if all(rate >= 0.5 for rate in detection_rates):
            f.write(f"The watermark detection rate remained above 50% for all tested crop sizes (down to {min(percentages)}%).\n")
            f.write(f"This suggests that the watermarking scheme is robust against the crop-resize attack within the tested range.\n\n")
        elif all(rate < 0.5 for rate in detection_rates):
            f.write(f"The watermark detection rate was below 50% for all tested crop sizes (up to {max(percentages)}%).\n")
            f.write(f"This suggests that the watermarking scheme is vulnerable to the crop-resize attack even with minimal cropping.\n\n")
    
    # Add recommendations
    f.write("Recommendations:\n")
    f.write("--------------\n")
    if threshold_found and threshold_value is not None:
        if threshold_value < 40:
            f.write("The watermarking scheme shows poor resilience to crop-resize attacks.\n")
            f.write("Consider implementing a more robust watermarking algorithm or exploring techniques that embed the watermark more deeply in the image semantics.\n")
        elif threshold_value < 70:
            f.write("The watermarking scheme shows moderate resilience to crop-resize attacks.\n")
            f.write("Consider optimizing the watermark embedding to improve robustness against geometric transformations.\n")
        else:
            f.write("The watermarking scheme shows good resilience to crop-resize attacks.\n")
            f.write("The watermark remains detectable even with significant cropping, indicating a robust implementation.\n")
    else:
        if all(rate >= 0.5 for rate in detection_rates):
            f.write("The watermarking scheme is highly robust against crop-resize attacks.\n")
            f.write("Consider testing with even smaller crop percentages to find the actual threshold.\n")
        else:
            f.write("The watermarking scheme is highly vulnerable to crop-resize attacks.\n")
            f.write("A significant redesign of the watermarking approach may be necessary to improve robustness.\n")

print(f"\nTesting complete. Results saved to {args.output_dir}")
print(f"Summary report saved to {report_path}")
if threshold_found and threshold_value is not None:
    print(f"Estimated watermark detection threshold: {threshold_value:.1f}% crop size")