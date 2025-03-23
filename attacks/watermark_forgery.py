"""
Black Box Watermark Forgery Attack

This script implements the attack described in the paper where:
1. Take a watermarked image
2. Run it through exact inversion process
3. Generate a new image with a different prompt but using the latent from the inversion
4. Verify if the watermark is still present in the new image

The attack uses a proxy model (different from the one used for watermarking) to generate the new image.
"""

"""
python watermark_forgery.py \
 --input_image results/prc_num_10_steps_50_fpr_1e-05_nowm_0/original_images/1.png \
 --new_prompt "sunset" \
 --exp_id prc_num_10_steps_50_fpr_1e-05_nowm_0 \
 --original_model_id stabilityai/stable-diffusion-2-1-base \
 --proxy_model_id runwayml/stable-diffusion-v1-5 \
 --output_dir forgery_results
"""

import argparse
import os
import pickle
import torch
from PIL import Image
from tqdm import tqdm
import random
from pathlib import Path

from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, exact_inversion, generate

parser = argparse.ArgumentParser('Watermark Forgery Attack Args')
parser.add_argument('--input_image', type=str, required=True, help='Path to watermarked input image')
parser.add_argument('--new_prompt', type=str, required=True, help='New prompt for generating the forged image')
parser.add_argument('--original_model_id', type=str, default='stabilityai/stable-diffusion-2-1-base', 
                    help='Model ID used for watermarking and inversion')
parser.add_argument('--proxy_model_id', type=str, default='stabilityai/stable-diffusion-3.5-medium', 
                    help='Proxy model ID used for generating the forged image')
parser.add_argument('--exp_id', type=str, required=True, 
                    help='Experiment ID for loading the watermark keys (e.g., prc_num_10_steps_50_fpr_1e-05_nowm_0)')
parser.add_argument('--inf_steps', type=int, default=50, help='Number of inference steps')
parser.add_argument('--output_dir', type=str, default='forgery_results', help='Output directory for forged images')
parser.add_argument('--var', type=float, default=1.5, help='Variance for PRC watermark detection')

args = parser.parse_args()

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

# Load the input image
input_img = Image.open(args.input_image)
input_filename = Path(args.input_image).stem

print(f"Step 1: Running exact inversion on the watermarked image...")
# Run exact inversion to get the latents
reversed_latents = exact_inversion(
    input_img,
    prompt='',  # Empty prompt for inversion
    test_num_inference_steps=args.inf_steps,
    inv_order=0,  # Using inv_order=0 as in decode.py
    pipe=original_pipe
)

print(f"Step 2: Generating new image with proxy model using the inverted latents...")
# Generate a new image with the proxy model using the inverted latents
forged_image, _, _ = generate(
    prompt=args.new_prompt,
    init_latents=reversed_latents,
    num_inference_steps=args.inf_steps,
    solver_order=1,
    pipe=proxy_pipe
)

# Save the forged image
forged_image_path = f"{args.output_dir}/{input_filename}_forged.png"
forged_image.save(forged_image_path)
print(f"Forged image saved to {forged_image_path}")

print(f"Step 3: Verifying if the watermark is still present in the forged image...")
# Run exact inversion on the forged image to check for watermark
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

# Check for watermark
detection_result = Detect(decoding_key, forged_reversed_prc)
decoding_result = (Decode(decoding_key, forged_reversed_prc) is not None)
combined_result = detection_result or decoding_result

print(f"Watermark detection results for forged image:")
print(f"Detection: {detection_result}")
print(f"Decoding: {decoding_result}")
print(f"Combined: {combined_result}")

# Also check the original image for comparison
print(f"Checking original image for watermark...")
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

print(f"Watermark detection results for original image:")
print(f"Detection: {original_detection}")
print(f"Decoding: {original_decoding}")
print(f"Combined: {original_combined}")

print(f"Attack summary:")
print(f"Original image: {args.input_image}")
print(f"New prompt: {args.new_prompt}")
print(f"Forged image: {forged_image_path}")
print(f"Original image watermark detected: {original_combined}")
print(f"Forged image watermark detected: {combined_result}")
print(f"Attack {'successful' if combined_result else 'failed'}") 