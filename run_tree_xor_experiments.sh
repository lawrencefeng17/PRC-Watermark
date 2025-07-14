#!/bin/bash

# Tree XOR Watermarking Experiment Script
# Tests different group sizes with multiple prompts and runs

set -e  # Exit on any error

# Configuration
GROUP_SIZES=(2 3 4)
NUM_TOKENS=1024
NUM_RUNS_PER_PROMPT=5
MODEL_ID="google/gemma-3-1b-it"
OUTPUT_BASE_DIR="tree_xor_experiments"

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Array of prompts to test
PROMPTS=(
    "Write an extensive, winding summary and analysis of the Brothers Karamazov. It should be over 4000 words."
    "Generate the components of a social media website which hosts short form media. Write a detailed design document for the website. Then, provide the full implementation. It should be over 4000 words."
    "Describe the process of photosynthesis and its importance to life on Earth. Discuss theories on the relationship between evolution and photosynthesis. It should be over 4000 words."
    "Write a story about a cat who meets a dog in a post-apocalyptic world. They form a bond and go on an adventure together. It should be over 4000 words."
    "Discuss methods for controlling the development of artificial intelligence. Then do analysis of the pros and cons of control. It should be over 4000 words."
)

echo "Starting Tree XOR Watermarking Experiments"
echo "Group sizes: ${GROUP_SIZES[@]}"
echo "Prompts: ${#PROMPTS[@]}"
echo "Runs per prompt: $NUM_RUNS_PER_PROMPT"
echo "Total experiments: $((${#GROUP_SIZES[@]} * ${#PROMPTS[@]} * NUM_RUNS_PER_PROMPT))"
echo "=================================="

# Counter for progress tracking
total_experiments=$((${#GROUP_SIZES[@]} * ${#PROMPTS[@]} * NUM_RUNS_PER_PROMPT))
current_experiment=0

# Loop through group sizes
for group_size in "${GROUP_SIZES[@]}"; do
    echo "Testing group size: $group_size"
    
    # Loop through prompts
    for prompt_idx in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$prompt_idx]}"
        echo "  Prompt $((prompt_idx + 1)): ${prompt:0:50}..."
        
        # Loop through runs for this prompt
        for run in $(seq 1 $NUM_RUNS_PER_PROMPT); do
            current_experiment=$((current_experiment + 1))
            echo "    Run $run/$NUM_RUNS_PER_PROMPT (Progress: $current_experiment/$total_experiments)"
            
            # Create unique experiment ID
            timestamp=$(date +%Y%m%d_%H%M%S_%N)
            exp_id="tree_xor_gs${group_size}_p${prompt_idx}_r${run}_${timestamp}"
            
            # Run the experiment
            python watermarking/run_watermarking.py \
                --methods tree_xor \
                --model_id "$MODEL_ID" \
                --prompt "$prompt" \
                --n $NUM_TOKENS \
                --group_size $group_size \
                --output_dir "$OUTPUT_BASE_DIR" \
                --top_k 20 \
                --debug \

            echo "    Completed: $exp_id"
            done
        done
    echo "Completed all runs for group size $group_size"
    echo ""
done

echo "=================================="
echo "All experiments completed!"
echo "Results saved in: $OUTPUT_BASE_DIR"
echo "Use analyze_tree_xor_results.py to analyze the results" 