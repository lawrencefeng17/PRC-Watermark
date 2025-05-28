import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
BASE_EXPERIMENT_DIR = "/raid/lawrence/substitution_rate_experiments"
ANALYSIS_OUTPUT_DIR_NAME = "analysis_results"

# --- Helper Functions ---
def find_json_files(base_dir):
    """Finds all token_detection_results.json files recursively."""
    json_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "token_detection_results.json":
                json_files.append(Path(root) / file)
    return json_files

def parse_experiment_path(json_path, base_experiment_path):
    """Parses category, prompt_id, and top_p from the file path."""
    try:
        relative_path = json_path.relative_to(base_experiment_path)
        parts = relative_path.parts
        # Expected structure: <category>/<prompt_id>/<top_p_id>/<timestamped_run_dir>/token_detection_results.json
        category = parts[0]
        prompt_id = parts[1]
        top_p_str = parts[2].replace("top_p_", "").replace("_", ".")
        top_p = float(top_p_str)
        return category, prompt_id, top_p
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse path {json_path}: {e}")
        return None, None, None

# --- Main Analysis Script ---
def analyze_experiments():
    base_dir = Path(BASE_EXPERIMENT_DIR)
    analysis_output_dir = base_dir / ANALYSIS_OUTPUT_DIR_NAME
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting analysis for experiments in: {base_dir}")
    print(f"Analysis results will be saved to: {analysis_output_dir}")

    json_files = find_json_files(base_dir)
    if not json_files:
        print("No 'token_detection_results.json' files found. Exiting.")
        return

    print(f"Found {len(json_files)} result files to analyze.")

    all_data = []
    for file_path in json_files:
        category, prompt_id, top_p = parse_experiment_path(file_path, base_dir)
        if category is None:
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Ensure necessary keys exist, provide defaults if not (e.g. for older files or failed runs)
            rejection_rate = data.get('rejection_rate')
            avg_entropy = data.get('average_pushforward_entropy')
            detection_result = data.get('detection_result')
            hamming_weight = data.get('hamming_weight')
            threshold = data.get('threshold')
            
            # Calculate detection margin, only if threshold and hamming_weight are numbers
            detection_margin = None
            if isinstance(threshold, (int, float)) and isinstance(hamming_weight, (int, float)):
                detection_margin = threshold - hamming_weight
            
            all_data.append({
                'category': category,
                'prompt_id': prompt_id,
                'top_p': top_p,
                'rejection_rate': rejection_rate,
                'avg_entropy': avg_entropy,
                'detection_result': detection_result,
                'hamming_weight': hamming_weight,
                'threshold': threshold,
                'detection_margin': detection_margin,
                'file_path': str(file_path) # For debugging if needed
            })
        except Exception as e:
            print(f"Error reading or parsing {file_path}: {e}")
            continue
            
    if not all_data:
        print("No valid data could be parsed from JSON files. Exiting.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(analysis_output_dir / "compiled_results.csv", index=False)
    print(f"Compiled results saved to {analysis_output_dir / 'compiled_results.csv'}")

    # --- Statistical Summaries ---
    print("\n--- Statistical Summaries ---")

    metrics_to_summarize = ['rejection_rate', 'avg_entropy', 'hamming_weight', 'detection_margin']

    grouping_levels = {
        "Global": None,
        "Per Category": ['category'],
        "Per Top-p": ['top_p'],
        "Per Category & Top-p": ['category', 'top_p']
    }

    for group_name, group_cols in grouping_levels.items():
        print(f"\n== {group_name} Stats ==")
        if group_cols:
            grouped = df.groupby(group_cols)
        else:
            grouped = df # For global stats, effectively group by nothing

        for metric in metrics_to_summarize:
            if metric in df.columns:
                # Ensure data is numeric and drop NaNs for calculations
                numeric_series = pd.to_numeric(df[metric] if not group_cols else grouped[metric].apply(lambda x: x), errors='coerce').dropna()
                if not numeric_series.empty:
                    if group_cols:
                        print(f"  Mean {metric}:\n{grouped[metric].mean(numeric_only=True)}")
                        print(f"  Std Dev {metric}:\n{grouped[metric].std(numeric_only=True)}")
                    else:
                        print(f"  Mean {metric}: {df[metric].mean(numeric_only=True):.4f}")
                        print(f"  Std Dev {metric}: {df[metric].std(numeric_only=True):.4f}")
                else:
                    print(f"  {metric}: Not enough valid data for statistics.")
            else:
                print(f"  Metric '{metric}' not found in data.")

        # Detection Success Rate
        if 'detection_result' in df.columns:
            if group_cols:
                # Convert boolean to numeric for aggregation
                df_temp = df.copy()
                df_temp['detection_result_numeric'] = pd.to_numeric(df_temp['detection_result'], errors='coerce').fillna(0)
                grouped_temp = df_temp.groupby(group_cols)
                success_rate = grouped_temp['detection_result_numeric'].mean()
                print(f"  Detection Success Rate:\n{success_rate}")
            else:
                success_rate = pd.to_numeric(df['detection_result'], errors='coerce').fillna(0).mean()
                print(f"  Detection Success Rate: {success_rate:.2%}")
        else:
            print("  Metric 'detection_result' not found.")

    # --- Plot Generation ---
    print("\n--- Generating Plots ---")
    sns.set_theme(style="whitegrid")

    plot_configs = [
        {'y': 'rejection_rate', 'title': 'Rejection Rate vs. Top-p', 'ylabel': 'Rejection Rate'},
        {'y': 'avg_entropy', 'title': 'Average Pushforward Entropy vs. Top-p', 'ylabel': 'Avg. Pushforward Entropy'},
        {'y': 'hamming_weight', 'title': 'Hamming Weight vs. Top-p', 'ylabel': 'Hamming Weight'},
        {'y': 'detection_margin', 'title': 'Detection Margin vs. Top-p', 'ylabel': 'Detection Margin (Threshold - Hamming Weight)'},
    ]

    for config in plot_configs:
        plt.figure(figsize=(12, 7))
        plot_df = df.copy()
        plot_df[config['y']] = pd.to_numeric(plot_df[config['y']], errors='coerce')
        plot_df.dropna(subset=[config['y']], inplace=True)

        if not plot_df.empty:
            sns.boxplot(x='top_p', y=config['y'], hue='category', data=plot_df, palette="Set2")
            plt.title(config['title'])
            plt.xlabel("Top-p")
            plt.ylabel(config['ylabel'])
            plt.legend(title='Prompt Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plot_path = analysis_output_dir / f"{config['y']}_vs_top_p_boxplot.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
        else:
            print(f"Skipping plot for {config['title']} due to no valid data after filtering/conversion.")

    # Bar chart for detection success rate (fixed to handle 0% success rates)
    if 'detection_result' in df.columns:
        plt.figure(figsize=(12, 7))
        # Ensure detection_result is boolean or 0/1 for mean calculation
        df_copy = df.copy()
        df_copy['detection_result_numeric'] = pd.to_numeric(df_copy['detection_result'], errors='coerce').fillna(0).astype(int)
        
        # Group by category and top_p to calculate mean success rate
        success_rate_df = df_copy.groupby(['category', 'top_p'])['detection_result_numeric'].mean().reset_index()
        success_rate_df.rename(columns={'detection_result_numeric': 'success_rate'}, inplace=True)

        if not success_rate_df.empty:
            # Create the bar plot
            ax = sns.barplot(x='top_p', y='success_rate', hue='category', data=success_rate_df, palette="Set2", dodge=True)
            plt.title('Detection Success Rate vs. Top-p')
            plt.xlabel("Top-p")
            plt.ylabel("Detection Success Rate")
            plt.ylim(0, 1)
            
            # Add percentage labels on bars
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.1%}', 
                               (p.get_x() + p.get_width()/2., height),
                               ha='center', va='bottom')
            
            plt.legend(title='Prompt Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plot_path = analysis_output_dir / "detection_success_rate_vs_top_p_barplot.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
        else:
            print("Skipping plot for Detection Success Rate due to no valid data.")
    
    # Scatter plot: Avg Entropy vs. Rejection Rate
    if 'avg_entropy' in df.columns and 'rejection_rate' in df.columns:
        plt.figure(figsize=(10, 6))
        df_scatter = df.copy()
        df_scatter['avg_entropy'] = pd.to_numeric(df_scatter['avg_entropy'], errors='coerce')
        df_scatter['rejection_rate'] = pd.to_numeric(df_scatter['rejection_rate'], errors='coerce')
        df_scatter.dropna(subset=['avg_entropy', 'rejection_rate'], inplace=True)
        
        if not df_scatter.empty:
            sns.scatterplot(x='avg_entropy', y='rejection_rate', hue='top_p', size='category', data=df_scatter, palette="viridis", alpha=0.7)
            plt.title('Average Pushforward Entropy vs. Rejection Rate')
            plt.xlabel("Average Pushforward Entropy")
            plt.ylabel("Rejection Rate")
            plt.legend(title='Top-p / Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plot_path = analysis_output_dir / "entropy_vs_rejection_scatter.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
        else:
            print("Skipping scatter plot for Entropy vs. Rejection Rate due to no valid data.")

    # New plot: Hamming Weight vs Detection Threshold comparison
    if 'hamming_weight' in df.columns and 'threshold' in df.columns:
        plt.figure(figsize=(12, 7))
        df_detection = df.copy()
        df_detection['hamming_weight'] = pd.to_numeric(df_detection['hamming_weight'], errors='coerce')
        df_detection['threshold'] = pd.to_numeric(df_detection['threshold'], errors='coerce')
        df_detection.dropna(subset=['hamming_weight', 'threshold'], inplace=True)
        
        if not df_detection.empty:
            # Create subplot to show both hamming weight and threshold
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot hamming weights
            sns.boxplot(x='top_p', y='hamming_weight', hue='category', data=df_detection, palette="Set2", ax=ax)
            
            # Add horizontal line for threshold (assuming threshold is constant)
            threshold_value = df_detection['threshold'].iloc[0]  # Get first threshold value
            ax.axhline(y=threshold_value, color='red', linestyle='--', linewidth=2, label=f'Detection Threshold ({threshold_value:.1f})')
            
            plt.title('Hamming Weight vs. Detection Threshold by Top-p')
            plt.xlabel("Top-p")
            plt.ylabel("Hamming Weight")
            plt.legend(title='Category / Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
            plot_path = analysis_output_dir / "hamming_weight_threshold_comparison.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
        else:
            print("Skipping plot for Hamming Weight vs Threshold due to no valid data.")

    print("\nAnalysis complete.")
    
    # Print summary insights
    total_experiments = len(df)
    successful_detections = pd.to_numeric(df['detection_result'], errors='coerce').fillna(0).sum()
    print(f"\n--- Summary Insights ---")
    print(f"Total experiments analyzed: {total_experiments}")
    print(f"Successful watermark detections: {int(successful_detections)} ({successful_detections/total_experiments:.1%})")
    
    if successful_detections == 0:
        print("\n⚠️  WARNING: No watermarks were successfully detected!")
        print("   This suggests that the watermarking scheme may need parameter adjustment.")
        avg_hamming = df['hamming_weight'].mean(numeric_only=True)
        avg_threshold = df['threshold'].mean(numeric_only=True)
        print(f"   Average Hamming Weight: {avg_hamming:.1f}")
        print(f"   Average Threshold: {avg_threshold:.1f}")
        print(f"   Hamming weights are {avg_hamming - avg_threshold:.1f} points above threshold on average.")

if __name__ == "__main__":
    analyze_experiments() 