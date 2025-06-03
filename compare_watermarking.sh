#!/bin/bash

# Array of top-p values to test
top_p_values=(0.9 0.95 0.98 0.99 1.00)

# Model and other fixed parameters
model_id="meta-llama/Llama-3.2-1B-Instruct"
prompt="Write a thrilling story about a murder investigation in an old mansion."
output_dir="/home/lawrence/PRC-Watermark/llama_degeneration_experiments"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

echo "Starting watermarking vs baseline comparison..."
echo "Output directory: $output_dir"
echo "=============================================="

for top_p in "${top_p_values[@]}"; do
    echo ""
    echo "Running experiments with top_p = $top_p"
    echo "----------------------------------------"
    
    echo "1. Running watermarked text generation..."
    
    # Run watermarking and capture output to find experiment directory
    watermark_output=$(python watermarking/run_watermarking.py \
        --model_id "$model_id" \
        --n 2048 \
        --prc_t 3 \
        --temperature 1 \
        --debug \
        --new \
        --top_p "$top_p" \
        --methods token \
        --output_dir "$output_dir" 2>&1)
    
    # Print watermarking output
    echo "$watermark_output"
    
    # Extract experiment directory from the output
    experiment_dir=$(echo "$watermark_output" | grep -o "Results saved to: .*" | sed 's/Results saved to: //')
    
    if [ -n "$experiment_dir" ]; then
        echo ""
        echo "Experiment directory: $experiment_dir"
        echo ""
        echo "2. Running baseline text generation..."
        
        # Run baseline and save to the same experiment directory
        python baselines/top_p_standalone.py \
            --model_id "$model_id" \
            --prompt "$prompt" \
            --top_p "$top_p" \
            -m 2048 \
            --output_dir "$experiment_dir"
        
        # Rename the baseline output to distinguish it from watermarked output
        if [ -f "$experiment_dir/generated_text.txt" ]; then
            mv "$experiment_dir/generated_text.txt" "$experiment_dir/baseline_output.txt"
            echo "Baseline text saved to: $experiment_dir/baseline_output.txt"
        fi
    else
        echo "Warning: Could not extract experiment directory from watermarking output"
        echo "Running baseline without output directory..."
        
        python baselines/top_p_standalone.py \
            --model_id "$model_id" \
            --prompt "$prompt" \
            --top_p "$top_p" \
            -m 2048
    fi
    
    echo ""
    echo "Completed experiments for top_p = $top_p"
    echo "----------------------------------------"
done

echo ""
echo "All experiments completed!"
echo "==========================" 