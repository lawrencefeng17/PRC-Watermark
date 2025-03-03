"""
Analyze Forgery Results

This script analyzes and visualizes the results of the batch forgery attack.
It generates plots and statistics to help understand the effectiveness of the attack.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser('Analyze Forgery Results')
parser.add_argument('--results_file', type=str, required=True, help='Path to the results JSON file')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for analysis results')
args = parser.parse_args()

# Load results
with open(args.results_file, 'r') as f:
    results = json.load(f)

# Set output directory
if args.output_dir is None:
    args.output_dir = os.path.dirname(args.results_file)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Extract basic information
original_model = results['original_model']
proxy_model = results['proxy_model']
exp_id = results['exp_id']
success_rate = results['success_rate']
total_images = len(results['images'])
successful_attacks = sum(1 for img in results['images'] if img['attack_successful'])

# Create a DataFrame for easier analysis
data = []
for img in results['images']:
    data.append({
        'original_image': os.path.basename(img['original_image']),
        'forged_image': os.path.basename(img['forged_image']),
        'new_prompt': img['new_prompt'],
        'original_watermark_detected': img['original_watermark_detected'],
        'forged_watermark_detected': img['forged_watermark_detected'],
        'attack_successful': img['attack_successful']
    })
df = pd.DataFrame(data)

# Print summary statistics
print(f"Analysis of Forgery Attack Results")
print(f"==================================")
print(f"Original Model: {original_model}")
print(f"Proxy Model: {proxy_model}")
print(f"Experiment ID: {exp_id}")
print(f"Total Images: {total_images}")
print(f"Successful Attacks: {successful_attacks}")
print(f"Success Rate: {success_rate:.2%}")
print(f"Original Watermark Detection Rate: {df['original_watermark_detected'].mean():.2%}")
print(f"Forged Watermark Detection Rate: {df['forged_watermark_detected'].mean():.2%}")

# Create a bar chart of success rates
plt.figure(figsize=(10, 6))
labels = ['Original Images', 'Forged Images']
values = [df['original_watermark_detected'].mean(), df['forged_watermark_detected'].mean()]
colors = ['blue', 'red']
plt.bar(labels, values, color=colors)
plt.ylim(0, 1.1)
plt.ylabel('Watermark Detection Rate')
plt.title(f'Watermark Detection Rates\nOriginal Model: {original_model}\nProxy Model: {proxy_model}')
for i, v in enumerate(values):
    plt.text(i, v + 0.05, f'{v:.2%}', ha='center')
plt.savefig(os.path.join(args.output_dir, 'detection_rates.png'), dpi=300, bbox_inches='tight')

# Create a pie chart of attack success rate
plt.figure(figsize=(8, 8))
labels = ['Successful', 'Failed']
sizes = [successful_attacks, total_images - successful_attacks]
colors = ['green', 'red']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title(f'Attack Success Rate\n{successful_attacks}/{total_images} ({success_rate:.2%})')
plt.savefig(os.path.join(args.output_dir, 'success_rate_pie.png'), dpi=300, bbox_inches='tight')

# Create a comparison grid of original and forged images (first 5 pairs)
max_display = min(5, total_images)
fig, axes = plt.subplots(max_display, 2, figsize=(12, 3*max_display))
for i in range(max_display):
    img_data = results['images'][i]
    
    # Original image
    orig_img = Image.open(img_data['original_image'])
    axes[i, 0].imshow(np.array(orig_img))
    axes[i, 0].set_title(f"Original (Watermark: {'Yes' if img_data['original_watermark_detected'] else 'No'})")
    axes[i, 0].axis('off')
    
    # Forged image
    forged_img = Image.open(img_data['forged_image'])
    axes[i, 1].imshow(np.array(forged_img))
    axes[i, 1].set_title(f"Forged (Watermark: {'Yes' if img_data['forged_watermark_detected'] else 'No'})")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'image_comparison.png'), dpi=300, bbox_inches='tight')

# Generate a detailed report
report_path = os.path.join(args.output_dir, 'analysis_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Forgery Attack Analysis Report\n")
    f.write(f"============================\n\n")
    f.write(f"Configuration:\n")
    f.write(f"- Original Model: {original_model}\n")
    f.write(f"- Proxy Model: {proxy_model}\n")
    f.write(f"- Experiment ID: {exp_id}\n\n")
    
    f.write(f"Results Summary:\n")
    f.write(f"- Total Images: {total_images}\n")
    f.write(f"- Successful Attacks: {successful_attacks}\n")
    f.write(f"- Success Rate: {success_rate:.2%}\n")
    f.write(f"- Original Watermark Detection Rate: {df['original_watermark_detected'].mean():.2%}\n")
    f.write(f"- Forged Watermark Detection Rate: {df['forged_watermark_detected'].mean():.2%}\n\n")
    
    f.write(f"Detailed Results:\n")
    for i, img in enumerate(results['images']):
        f.write(f"Image {i+1}:\n")
        f.write(f"- Original Image: {os.path.basename(img['original_image'])}\n")
        f.write(f"- Forged Image: {os.path.basename(img['forged_image'])}\n")
        f.write(f"- New Prompt: {img['new_prompt']}\n")
        f.write(f"- Original Watermark Detected: {img['original_watermark_detected']}\n")
        f.write(f"- Forged Watermark Detected: {img['forged_watermark_detected']}\n")
        f.write(f"- Attack Successful: {img['attack_successful']}\n\n")
    
    f.write(f"Conclusion:\n")
    if success_rate > 0.8:
        f.write(f"The attack was highly successful with a {success_rate:.2%} success rate, indicating a significant vulnerability in the watermarking scheme.\n")
    elif success_rate > 0.5:
        f.write(f"The attack was moderately successful with a {success_rate:.2%} success rate, suggesting some vulnerability in the watermarking scheme.\n")
    else:
        f.write(f"The attack had limited success with a {success_rate:.2%} success rate, suggesting the watermarking scheme is relatively robust against this attack.\n")

print(f"\nAnalysis complete. Results saved to {args.output_dir}")
print(f"Report saved to {report_path}")
print(f"Visualizations saved to {args.output_dir}") 