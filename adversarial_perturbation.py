#!/usr/bin/env python3
"""
Script to find adversarial perturbations that break watermark detection
while maintaining visual quality of the image.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim

# Import from existing codebase
from inversion import stable_diffusion_pipe, exact_inversion
from plot_cosine_similarity import cosine_similarity
import src.pseudogaussians as prc_gaussians
from src.prc import Detect

# Set up argument parser
parser = argparse.ArgumentParser('Adversarial Perturbation Search')
parser.add_argument('-f', '--file', type=str, required=True, help='Path to the watermarked image')
parser.add_argument('--output_dir', type=str, default='adversarial_results', help='Directory to save results')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization')
parser.add_argument('--iterations', type=int, default=1000, help='Maximum number of iterations')
parser.add_argument('--l2_weight', type=float, default=0.1, help='Weight for L2 norm penalty')
parser.add_argument('--ssim_weight', type=float, default=0.5, help='Weight for SSIM penalty')
parser.add_argument('--detection_threshold', type=float, default=0.5, help='Cosine similarity threshold for detection')
parser.add_argument('--exp_id', type=str, default='prc_num_10_steps_50_fpr_1e-05_nowm_0', 
                    help='Experiment ID for loading keys')
args = parser.parse_args()

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def transform_img_to_tensor(img):
    """Convert PIL image to normalized tensor"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    return transform(img).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    """Convert normalized tensor back to PIL image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor.clone().detach().cpu().squeeze() + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    return T.ToPILImage()(tensor)

def compute_ssim(img1, img2):
    """Compute SSIM between two PIL images"""
    img1_np = np.array(img1.convert('L'))  # Convert to grayscale
    img2_np = np.array(img2.convert('L'))
    return ssim(img1_np, img2_np)

def find_adversarial_perturbation(image_path, pipe, decoding_key, 
                                 learning_rate=0.01, 
                                 max_iterations=1000,
                                 l2_weight=0.1,
                                 ssim_weight=0.5,
                                 detection_threshold=0.5,
                                 output_dir='adversarial_results'):
    """
    Find an adversarial perturbation that breaks watermark detection
    while maintaining visual quality.
    
    Args:
        image_path: Path to the watermarked image
        pipe: Stable diffusion pipeline
        decoding_key: Key for watermark detection
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of iterations
        l2_weight: Weight for L2 norm penalty
        ssim_weight: Weight for SSIM penalty
        detection_threshold: Cosine similarity threshold for detection
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results including:
        - original_image: PIL image
        - perturbed_image: PIL image
        - perturbation: Tensor
        - original_latent: Tensor
        - perturbed_latent: Tensor
        - cosine_similarity: Float
        - l2_norm: Float
        - ssim: Float
        - iterations: Int
        - success: Bool (whether detection was broken)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    original_image = Image.open(image_path)
    img_tensor = transform_img_to_tensor(original_image)
    
    # Get the original latent representation
    print("Computing original latent representation...")
    original_latent = exact_inversion(original_image, prompt='', pipe=pipe)
    
    # Check if the original image is detected as watermarked
    reversed_prc = prc_gaussians.recover_posteriors(
        original_latent.to(torch.float64).flatten().cpu(), 
        variances=1.5
    ).flatten().cpu()
    
    original_detection = Detect(decoding_key, reversed_prc)
    if not original_detection:
        print("Warning: Original image not detected as watermarked!")
        return None
    
    # Initialize perturbation
    perturbation = torch.zeros_like(img_tensor, requires_grad=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([perturbation], lr=learning_rate)
    
    # For tracking progress
    best_similarity = 1.0
    best_perturbation = None
    best_perturbed_img = None
    best_perturbed_latent = None
    best_iteration = 0
    best_l2_norm = float('inf')
    best_ssim_value = 0.0
    
    # For plotting
    iterations_list = []
    similarity_list = []
    l2_norm_list = []
    ssim_list = []
    
    print(f"Starting optimization for {max_iterations} iterations...")
    for i in tqdm(range(max_iterations)):
        # Apply perturbation
        perturbed_img_tensor = torch.clamp(img_tensor + perturbation, -1, 1)
        
        # Convert to PIL for inversion
        perturbed_pil = tensor_to_pil(perturbed_img_tensor)
        
        # Get latent representation of perturbed image
        perturbed_latent = exact_inversion(perturbed_pil, prompt='', pipe=pipe)
        
        # Compute cosine similarity
        similarity = cosine_similarity(original_latent, perturbed_latent)
        
        # Compute L2 norm of perturbation
        l2_norm = torch.norm(perturbation)
        
        # Compute SSIM between original and perturbed
        ssim_value = compute_ssim(original_image, perturbed_pil)
        
        # Total loss: minimize similarity, minimize perturbation size, maximize SSIM
        loss = similarity + l2_weight * l2_norm - ssim_weight * ssim_value
        
        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        iterations_list.append(i)
        similarity_list.append(similarity.item())
        l2_norm_list.append(l2_norm.item())
        ssim_list.append(ssim_value)
        
        # Check if this is the best result so far
        if similarity.item() < best_similarity:
            best_similarity = similarity.item()
            best_perturbation = perturbation.clone().detach()
            best_perturbed_img = perturbed_pil
            best_perturbed_latent = perturbed_latent.clone().detach()
            best_iteration = i
            best_l2_norm = l2_norm.item()
            best_ssim_value = ssim_value
            
            # Save intermediate result
            if i % 100 == 0:
                perturbed_pil.save(f"{output_dir}/perturbed_{Path(image_path).stem}_iter_{i}.png")
                
                # Visualize perturbation
                perturbation_vis = tensor_to_pil((perturbation - perturbation.min()) / 
                                               (perturbation.max() - perturbation.min() + 1e-8))
                perturbation_vis.save(f"{output_dir}/perturbation_{Path(image_path).stem}_iter_{i}.png")
        
        # Check if we've broken detection
        if similarity.item() < detection_threshold:
            print(f"Detection broken at iteration {i}!")
            
            # Verify with actual detection
            reversed_prc = prc_gaussians.recover_posteriors(
                perturbed_latent.to(torch.float64).flatten().cpu(), 
                variances=1.5
            ).flatten().cpu()
            
            detection_result = Detect(decoding_key, reversed_prc)
            
            if not detection_result:
                print("Confirmed: Watermark detection broken!")
                break
    
    # Apply best perturbation
    final_perturbed_img_tensor = torch.clamp(img_tensor + best_perturbation, -1, 1)
    final_perturbed_pil = tensor_to_pil(final_perturbed_img_tensor)
    
    # Save final results
    original_image.save(f"{output_dir}/original_{Path(image_path).stem}.png")
    final_perturbed_pil.save(f"{output_dir}/final_perturbed_{Path(image_path).stem}.png")
    
    # Visualize perturbation
    perturbation_vis = tensor_to_pil((best_perturbation - best_perturbation.min()) / 
                                   (best_perturbation.max() - best_perturbation.min() + 1e-8))
    perturbation_vis.save(f"{output_dir}/final_perturbation_{Path(image_path).stem}.png")
    
    # Plot metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(iterations_list, similarity_list)
    plt.axhline(y=detection_threshold, color='r', linestyle='--', label=f'Threshold: {detection_threshold}')
    plt.xlabel('Iteration')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs. Iteration')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(iterations_list, l2_norm_list)
    plt.xlabel('Iteration')
    plt.ylabel('L2 Norm')
    plt.title('Perturbation L2 Norm vs. Iteration')
    
    plt.subplot(1, 3, 3)
    plt.plot(iterations_list, ssim_list)
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.title('SSIM vs. Iteration')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_{Path(image_path).stem}.png")
    
    # Final verification
    final_latent = exact_inversion(final_perturbed_pil, prompt='', pipe=pipe)
    reversed_prc = prc_gaussians.recover_posteriors(
        final_latent.to(torch.float64).flatten().cpu(), 
        variances=1.5
    ).flatten().cpu()
    
    final_detection = Detect(decoding_key, reversed_prc)
    final_similarity = cosine_similarity(original_latent, final_latent).item()
    
    print(f"Final results:")
    print(f"  Cosine similarity: {final_similarity:.4f}")
    print(f"  L2 norm of perturbation: {best_l2_norm:.4f}")
    print(f"  SSIM: {best_ssim_value:.4f}")
    print(f"  Detection broken: {not final_detection}")
    
    return {
        'original_image': original_image,
        'perturbed_image': final_perturbed_pil,
        'perturbation': best_perturbation,
        'original_latent': original_latent,
        'perturbed_latent': final_latent,
        'cosine_similarity': final_similarity,
        'l2_norm': best_l2_norm,
        'ssim': best_ssim_value,
        'iterations': best_iteration,
        'success': not final_detection
    }

def main():
    # Load the stable diffusion pipeline
    print("Loading stable diffusion pipeline...")
    pipe = stable_diffusion_pipe(solver_order=1)
    pipe.set_progress_bar_config(disable=True)
    
    # Load the decoding key
    print(f"Loading decoding key from experiment {args.exp_id}...")
    import pickle
    with open(f'keys/{args.exp_id}.pkl', 'rb') as f:
        _, decoding_key = pickle.load(f)
    
    # Find adversarial perturbation
    results = find_adversarial_perturbation(
        args.file, 
        pipe, 
        decoding_key,
        learning_rate=args.lr,
        max_iterations=args.iterations,
        l2_weight=args.l2_weight,
        ssim_weight=args.ssim_weight,
        detection_threshold=args.detection_threshold,
        output_dir=args.output_dir
    )
    
    if results:
        # Save results summary
        with open(f"{args.output_dir}/results_{Path(args.file).stem}.txt", 'w') as f:
            f.write(f"Image: {args.file}\n")
            f.write(f"Cosine similarity: {results['cosine_similarity']:.4f}\n")
            f.write(f"L2 norm of perturbation: {results['l2_norm']:.4f}\n")
            f.write(f"SSIM: {results['ssim']:.4f}\n")
            f.write(f"Iterations: {results['iterations']}\n")
            f.write(f"Detection broken: {results['success']}\n")

if __name__ == "__main__":
    main() 