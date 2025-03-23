"""
Batch Black Box Watermark Forgery Attack

This script implements the batch version of the watermark forgery attack where:
1. Take multiple watermarked images from a directory
2. Run each through exact inversion process
3. Generate new images with different prompts but using the latents from the inversion
4. Verify if the watermarks are still present in the new images
"""

import argparse
import os
import pickle
import torch
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, exact_inversion, generate

parser = argparse.ArgumentParser('Batch Watermark Forgery Attack Args')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing watermarked input images')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts', 
                    help='Dataset ID for prompts')
parser.add_argument('--original_model_id', type=str, default='stabilityai/stable-diffusion-2-1-base', 
                    help='Model ID used for watermarking and inversion')
parser.add_argument('--proxy_model_id', type=str, default='runwayml/stable-diffusion-v1-5', 
                    help='Proxy model ID used for generating the forged image')
parser.add_argument('--exp_id', type=str, required=True, 
                    help='Experiment ID for loading the watermark keys (e.g., prc_num_10_steps_50_fpr_1e-05_nowm_0)')
parser.add_argument('--inf_steps', type=int, default=50, help='Number of inference steps')
parser.add_argument('--output_dir', type=str, default='batch_forgery_results', help='Output directory for forged images')
parser.add_argument('--var', type=float, default=1.5, help='Variance for PRC watermark detection')
parser.add_argument('--num_images', type=int, default=5, help='Number of images to process')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

args = parser.parse_args()

# Set random seed for reproducibility
def seed_everything(seed):
    random.seed(seed)
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
hf_cache_dir = '/home/lawrence/.cache/huggingface/hub'

# Load watermark keys
with open(f'keys/{args.exp_id}.pkl', 'rb') as f:
    encoding_key, decoding_key = pickle.load(f)
    print(f'Loaded PRC keys from file keys/{args.exp_id}.pkl')

# Load the original model for inversion
original_pipe = stable_diffusion_pipe(solver_order=1, model_id=args.original_model_id, cache_dir=hf_cache_dir)
original_pipe.set_progress_bar_config(disable=True)

# Load the proxy model for generation
proxy_pipe = stable_diffusion_pipe(solver_order=1, model_id=args.proxy_model_id, cache_dir=hf_cache_dir)
proxy_pipe.set_progress_bar_config(disable=True)

# Get list of image files in the input directory
image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
if args.num_images < len(image_files):
    image_files = random.sample(image_files, args.num_images)

# Load prompts from dataset
if args.dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']]

# Sample random prompts
new_prompts = random.sample(all_prompts, len(image_files))

# Create a results dictionary
results = {
    "original_model": args.original_model_id,
    "proxy_model": args.proxy_model_id,
    "exp_id": args.exp_id,
    "images": []
}

# Process each image
for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
    image_path = os.path.join(args.input_dir, image_file)
    input_img = Image.open(image_path)
    input_filename = Path(image_file).stem
    new_prompt = new_prompts[i]
    
    # Step 1: Run exact inversion on the watermarked image
    reversed_latents = exact_inversion(
        input_img,
        prompt='',  # Empty prompt for inversion
        test_num_inference_steps=args.inf_steps,
        inv_order=0,
        pipe=original_pipe
    )
    
    # Step 2: Generate new image with proxy model using the inverted latents
    forged_image, _, _ = generate(
        prompt=new_prompt,
        init_latents=reversed_latents,
        num_inference_steps=args.inf_steps,
        solver_order=1,
        pipe=proxy_pipe
    )
    
    # Save the forged image
    forged_image_path = f"{args.output_dir}/{input_filename}_forged.png"
    forged_image.save(forged_image_path)
    
    # Step 3: Verify if the watermark is still present in the forged image
    forged_reversed_latents = exact_inversion(
        forged_image,
        prompt='',
        test_num_inference_steps=args.inf_steps,
        inv_order=0,
        pipe=original_pipe
    )
    
    # Recover posteriors for watermark detection
    forged_reversed_prc = prc_gaussians.recover_posteriors(
        forged_reversed_latents.to(torch.float64).flatten().cpu(), 
        variances=float(args.var)
    ).flatten().cpu()
    
    # Save the recovered posteriors
    torch.save(forged_reversed_prc, f'{args.output_dir}/{input_filename}_forged.pt')
    
    # Check for watermark in forged image
    forged_detection = Detect(decoding_key, forged_reversed_prc)
    forged_decoding = (Decode(decoding_key, forged_reversed_prc) is not None)
    forged_combined = forged_detection or forged_decoding
    
    # Check original image for watermark
    original_reversed_latents = exact_inversion(
        input_img,
        prompt='',
        test_num_inference_steps=args.inf_steps,
        inv_order=0,
        pipe=original_pipe
    )
    
    original_reversed_prc = prc_gaussians.recover_posteriors(
        original_reversed_latents.to(torch.float64).flatten().cpu(), 
        variances=float(args.var)
    ).flatten().cpu()
    
    original_detection = Detect(decoding_key, original_reversed_prc)
    original_decoding = (Decode(decoding_key, original_reversed_prc) is not None)
    original_combined = original_detection or original_decoding
    
    # Store results
    image_result = {
        "original_image": image_path,
        "forged_image": forged_image_path,
        "new_prompt": new_prompt,
        "original_watermark_detected": bool(original_combined),
        "forged_watermark_detected": bool(forged_combined),
        "attack_successful": bool(forged_combined)
    }
    results["images"].append(image_result)
    
    print(f"Image {i+1}/{len(image_files)}: {image_file}")
    print(f"  New prompt: {new_prompt}")
    print(f"  Original watermark detected: {original_combined}")
    print(f"  Forged watermark detected: {forged_combined}")
    print(f"  Attack {'successful' if forged_combined else 'failed'}")

# Calculate success rate
success_count = sum(1 for img in results["images"] if img["attack_successful"])
success_rate = success_count / len(results["images"]) if results["images"] else 0
results["success_rate"] = success_rate

# Save results to JSON file
with open(f"{args.output_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nProcessed {len(results['images'])} images")
print(f"Attack success rate: {success_rate:.2%} ({success_count}/{len(results['images'])})")
print(f"Results saved to {args.output_dir}/results.json") 