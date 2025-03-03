# Adversarial Perturbation Analysis for Watermarked Images

This set of scripts allows you to find and analyze adversarial perturbations that can break watermark detection while maintaining visual quality of the images.

## Overview

The watermarking scheme embeds signals in the latent space during the diffusion process. These scripts help you understand the robustness of this scheme by finding minimal perturbations in the image space that break the watermark detection.

The approach uses gradient-based optimization to find image-specific perturbations that:
1. Minimize the cosine similarity between the original latent and the perturbed latent
2. Maintain visual quality (high SSIM)
3. Have minimal L2 norm (small perturbation)

## Scripts

### 1. `adversarial_perturbation.py`

This script finds an adversarial perturbation for a single watermarked image.

```bash
python adversarial_perturbation.py -f path/to/watermarked_image.png [options]
```

Options:
- `--output_dir`: Directory to save results (default: 'adversarial_results')
- `--lr`: Learning rate for optimization (default: 0.01)
- `--iterations`: Maximum number of iterations (default: 1000)
- `--l2_weight`: Weight for L2 norm penalty (default: 0.1)
- `--ssim_weight`: Weight for SSIM penalty (default: 0.5)
- `--detection_threshold`: Cosine similarity threshold for detection (default: 0.5)
- `--exp_id`: Experiment ID for loading keys (default: 'prc_num_10_steps_50_fpr_1e-05_nowm_0')

### 2. `batch_adversarial.py`

This script runs adversarial perturbation search on multiple watermarked images.

```bash
python batch_adversarial.py --images_dir path/to/watermarked_images [options]
```

Options:
- `--output_dir`: Directory to save results (default: 'adversarial_results')
- `--exp_id`: Experiment ID for loading keys (default: 'prc_num_10_steps_50_fpr_1e-05_nowm_0')
- `--lr`: Learning rate for optimization (default: 0.01)
- `--iterations`: Maximum number of iterations (default: 1000)
- `--l2_weight`: Weight for L2 norm penalty (default: 0.1)
- `--ssim_weight`: Weight for SSIM penalty (default: 0.5)
- `--detection_threshold`: Cosine similarity threshold for detection (default: 0.5)
- `--max_workers`: Maximum number of parallel workers (default: 1)
- `--file_pattern`: Pattern to match image files (default: '*.png')
- `--analyze`: Run analysis after completion (flag)

### 3. `analyze_perturbations.py`

This script analyzes the results from multiple adversarial perturbation runs to identify patterns.

```bash
python analyze_perturbations.py [options]
```

Options:
- `--results_dir`: Directory containing adversarial perturbation results (default: 'adversarial_results')
- `--output_dir`: Directory to save analysis results (default: 'perturbation_analysis')
- `--n_components`: Number of PCA components to extract (default: 10)
- `--n_clusters`: Number of clusters for K-means (default: 3)

## Example Workflow

1. Generate watermarked images (using your existing code)
2. Run batch adversarial perturbation search:
   ```bash
   python batch_adversarial.py --images_dir results/prc_num_10_steps_50_fpr_1e-05_nowm_0/original_images --analyze
   ```
3. Examine the results in the `adversarial_results` and `adversarial_results_analysis` directories

## Output Files

For each image, the following files are generated:

- `original_[image_name].png`: The original watermarked image
- `final_perturbed_[image_name].png`: The perturbed image that breaks watermark detection
- `final_perturbation_[image_name].png`: Visualization of the perturbation
- `metrics_[image_name].png`: Plot of metrics during optimization
- `results_[image_name].txt`: Summary of results

The analysis generates:

- `metrics_analysis.png`: Analysis of metrics across all images
- `pca_variance.png`: PCA explained variance
- `pca_components.png`: Visualization of top principal components
- `perturbation_clusters.png`: Clustering of perturbations
- `cluster_analysis.txt`: Analysis of cluster characteristics
- `overall_analysis.txt`: Overall analysis of results

## Interpreting Results

- **Cosine Similarity**: Lower values indicate more effective perturbations
- **L2 Norm**: Lower values indicate smaller perturbations
- **SSIM**: Higher values indicate better visual quality
- **Success Rate**: Percentage of images where watermark detection was broken

The PCA analysis helps identify common patterns in successful perturbations, which could reveal weaknesses in the watermarking scheme.

## Requirements

- PyTorch
- PIL
- NumPy
- Matplotlib
- scikit-learn
- scikit-image
- tqdm

## Notes

- GPU memory usage can be high during the inversion process. If you encounter memory issues, reduce batch sizes or use sequential processing.
- The optimization process can be time-consuming. Start with a small number of iterations and increase as needed.
- Experiment with different weights for L2 norm and SSIM to find the optimal trade-off between effectiveness and visual quality. 