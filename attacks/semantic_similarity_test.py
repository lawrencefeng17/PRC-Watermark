"""
Semantic Similarity Watermark Forgery Test

This script tests how semantic similarity between prompts affects watermark forgery success.
It uses sentence transformers to compute prompt similarity and analyzes correlation with:
1. Watermark transfer success
2. Latent space cosine similarity
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
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
import pandas as pd

from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians
from inversion import stable_diffusion_pipe, exact_inversion, generate

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    return torch.nn.functional.cosine_similarity(v1.flatten(), v2.flatten(), dim=0)

def get_semantic_similarity(prompt1, prompt2, text_model):
    """Compute semantic similarity between two prompts using sentence transformers"""
    emb1 = text_model.encode(prompt1, convert_to_tensor=True)
    emb2 = text_model.encode(prompt2, convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

def plot_correlation(x, y, xlabel, ylabel, title, output_path):
    """Create a scatter plot with correlation line"""
    plt.figure(figsize=(10, 6))
    
    # Calculate correlation coefficient
    correlation, p_value = pearsonr(x, y)
    
    # Create scatter plot with regression line
    sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5})
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nCorrelation: {correlation:.3f} (p={p_value:.3f})")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation, p_value

def main():
    parser = argparse.ArgumentParser('Semantic Similarity Watermark Forgery Test')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing watermarked input images')
    parser.add_argument('--original_prompts_file', type=str, required=True, help='JSON file containing original prompts')
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID for loading watermark keys')
    parser.add_argument('--original_model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--proxy_model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--num_variations', type=int, default=5, 
                        help='Number of semantic variations to test per image')
    parser.add_argument('--inf_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='semantic_similarity_results')
    parser.add_argument('--var', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load sentence transformer for semantic similarity
    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load original prompts
    with open(args.original_prompts_file, 'r') as f:
        original_prompts = json.load(f)

    # Load watermark keys
    with open(f'keys/{args.exp_id}.pkl', 'rb') as f:
        encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{args.exp_id}.pkl')

    # Initialize models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hf_cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    
    original_pipe = stable_diffusion_pipe(solver_order=1, model_id=args.original_model_id, cache_dir=hf_cache_dir)
    original_pipe.set_progress_bar_config(disable=True)
    
    proxy_pipe = stable_diffusion_pipe(solver_order=1, model_id=args.proxy_model_id, cache_dir=hf_cache_dir)
    proxy_pipe.set_progress_bar_config(disable=True)

    # Get list of image files
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Prepare results dictionary
    results = {
        "original_model": args.original_model_id,
        "proxy_model": args.proxy_model_id,
        "exp_id": args.exp_id,
        "tests": []
    }

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.input_dir, image_file)
        input_img = Image.open(image_path)
        input_filename = Path(image_file).stem
        original_prompt = original_prompts[input_filename]

        # Get original image latents and PRC
        original_latents = exact_inversion(
            input_img,
            prompt='',
            test_num_inference_steps=args.inf_steps,
            inv_order=0,
            pipe=original_pipe
        )
        
        original_prc = prc_gaussians.recover_posteriors(
            original_latents.to(torch.float64).flatten().cpu(),
            variances=float(args.var)
        ).flatten().cpu()

        # Verify original watermark
        original_detection = Detect(decoding_key, original_prc)
        if not original_detection:
            print(f"Warning: No watermark detected in original image {image_file}")
            continue

        # Generate semantic variations
        variations = []
        
        # Very similar prompt (style variation)
        variations.append(original_prompt + ", in the style of oil painting")
        
        # Somewhat similar prompt (content preserved, different style)
        variations.append(original_prompt.replace("digital art", "watercolor painting").replace("artstation", "traditional art"))
        
        # Different but related prompt (similar subject, different context)
        base_subject = original_prompt.split(",")[0]
        variations.append(f"{base_subject}, in a cyberpunk city at night, neon lights")
        
        # Very different prompt (different subject, different style)
        variations.append("a serene landscape with mountains and a lake at sunset, bob ross style")
        
        # Completely different prompt
        variations.append("a close-up macro photograph of a butterfly on a flower")

        # Test each variation
        for new_prompt in tqdm(variations, desc=f"Testing variations for {image_file}", leave=False):
            # Generate forged image
            forged_image, _, forged_latents = generate(
                prompt=new_prompt,
                init_latents=original_latents,
                num_inference_steps=args.inf_steps,
                solver_order=1,
                pipe=proxy_pipe
            )

            # Save forged image
            forged_image_path = os.path.join(args.output_dir, f"{input_filename}_{len(results['tests'])}_forged.png")
            forged_image.save(forged_image_path)

            # Get forged image PRC
            forged_prc = prc_gaussians.recover_posteriors(
                forged_latents.to(torch.float64).flatten().cpu(),
                variances=float(args.var)
            ).flatten().cpu()

            # Check watermark in forged image
            forged_detection = Detect(decoding_key, forged_prc)

            # Calculate similarities
            semantic_similarity = get_semantic_similarity(original_prompt, new_prompt, text_model)
            latent_similarity = compute_cosine_similarity(original_latents, forged_latents).item()
            prc_similarity = compute_cosine_similarity(original_prc, forged_prc).item()

            # Store results
            test_result = {
                "original_image": image_path,
                "forged_image": forged_image_path,
                "original_prompt": original_prompt,
                "new_prompt": new_prompt,
                "semantic_similarity": semantic_similarity,
                "latent_similarity": latent_similarity,
                "prc_similarity": prc_similarity,
                "watermark_preserved": bool(forged_detection)
            }
            results["tests"].append(test_result)

    # Save full results
    with open(os.path.join(args.output_dir, "semantic_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Create analysis dataframe
    df = pd.DataFrame(results["tests"])

    # Calculate success rates
    overall_success_rate = df["watermark_preserved"].mean()
    print(f"\nOverall watermark preservation rate: {overall_success_rate:.2%}")

    # Create correlation plots
    metrics = [
        ("semantic_similarity", "Prompt Semantic Similarity"),
        ("latent_similarity", "Latent Space Similarity"),
        ("prc_similarity", "PRC Vector Similarity")
    ]

    correlations = {}
    for metric, metric_name in metrics:
        # Plot correlation with watermark preservation
        x = df[metric]
        y = df["watermark_preserved"].astype(float)
        
        correlation, p_value = plot_correlation(
            x, y,
            xlabel=metric_name,
            ylabel="Watermark Preserved",
            title=f"Watermark Preservation vs {metric_name}",
            output_path=os.path.join(args.output_dir, f"{metric}_vs_preservation.png")
        )
        
        correlations[metric] = {"correlation": correlation, "p_value": p_value}

    # Create correlation matrix between all metrics
    correlation_matrix = df[["semantic_similarity", "latent_similarity", "prc_similarity"]].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Similarity Metrics")
    plt.savefig(os.path.join(args.output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Generate summary report
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("Semantic Similarity Watermark Forgery Analysis\n")
        f.write("==========================================\n\n")
        
        f.write("Configuration:\n")
        f.write(f"- Original Model: {args.original_model_id}\n")
        f.write(f"- Proxy Model: {args.proxy_model_id}\n")
        f.write(f"- Experiment ID: {args.exp_id}\n")
        f.write(f"- Number of tests: {len(results['tests'])}\n\n")
        
        f.write("Results Summary:\n")
        f.write(f"- Overall watermark preservation rate: {overall_success_rate:.2%}\n\n")
        
        f.write("Correlation Analysis:\n")
        for metric, metric_name in metrics:
            corr = correlations[metric]
            f.write(f"\n{metric_name}:\n")
            f.write(f"- Correlation with watermark preservation: {corr['correlation']:.3f}\n")
            f.write(f"- P-value: {corr['p_value']:.3f}\n")
        
        f.write("\nConclusions:\n")
        
        # Add conclusions based on correlation strengths
        strongest_correlation = max(correlations.items(), key=lambda x: abs(x[1]["correlation"]))
        f.write(f"- The strongest correlation was found with {dict(metrics)[strongest_correlation[0]]}\n")
        
        if abs(correlations["semantic_similarity"]["correlation"]) > 0.3:
            f.write("- Semantic similarity shows a significant relationship with watermark preservation\n")
        else:
            f.write("- Semantic similarity shows weak relationship with watermark preservation\n")
            
        if abs(correlations["latent_similarity"]["correlation"]) > 0.3:
            f.write("- Latent space similarity is a good predictor of watermark preservation\n")
        else:
            f.write("- Latent space similarity is not a strong predictor of watermark preservation\n")

if __name__ == "__main__":
    main() 