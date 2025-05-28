#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <TOP_P> <PROMPT> <BASE_OUTPUT_DIR>"
    exit 1
fi

# Assign arguments to variables
TOP_P=$1
PROMPT=$2
BASE_OUTPUT_DIR=$3

# Configurable parameters
MODEL_ID="google/gemma-3-1b-it"
TEMPERATURE=1
NUM_TOKENS=1024
N_BITS=1024

# Create the base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Run watermarking script
echo "Running watermarking script..."
PYTHON_CMD="python watermarking/run_watermarking.py \
    --top_p $TOP_P \
    --debug \
    --new \
    --methods token \
    --prompt \"$PROMPT\" \
    --n $N_BITS \
    --num_tokens $NUM_TOKENS \
    --temperature $TEMPERATURE \
    --model_id \"$MODEL_ID\" \
    --output_dir \"$BASE_OUTPUT_DIR\""

echo "Executing: $PYTHON_CMD"

# Execute and capture stdout and stderr separately
WATERMARK_OUTPUT_STDERR_FILE=$(mktemp) # Temporary file for stderr
WATERMARK_OUTPUT_STDOUT_FILE=$(mktemp) # Temporary file for stdout

if eval "$PYTHON_CMD" 1> "$WATERMARK_OUTPUT_STDOUT_FILE" 2> "$WATERMARK_OUTPUT_STDERR_FILE"; then
    echo "Watermarking script completed successfully (stdout below)."
    # WATERMARK_OUTPUT will be the stdout content for extracting experiment dir
    WATERMARK_OUTPUT=$(cat "$WATERMARK_OUTPUT_STDOUT_FILE")
    # Print stderr if any, even on success, for warnings
    if [ -s "$WATERMARK_OUTPUT_STDERR_FILE" ]; then
        echo "Watermarking script stderr (even on success):"
        cat "$WATERMARK_OUTPUT_STDERR_FILE"
    fi
else
    echo "ERROR: Watermarking script failed. Exit code: $?"
    echo "-------------------- Watermarking script STDOUT: --------------------"
    cat "$WATERMARK_OUTPUT_STDOUT_FILE"
    echo "-------------------- Watermarking script STDERR: --------------------"
    cat "$WATERMARK_OUTPUT_STDERR_FILE"
    echo "---------------------------------------------------------------------"
    rm -f "$WATERMARK_OUTPUT_STDERR_FILE" "$WATERMARK_OUTPUT_STDOUT_FILE"
    exit 1 # Exit run_comparison.sh if watermarking script fails
fi

rm -f "$WATERMARK_OUTPUT_STDERR_FILE" "$WATERMARK_OUTPUT_STDOUT_FILE"

# Extract experiment directory from the stdout
EXPERIMENT_DIR=$(echo "$WATERMARK_OUTPUT" | grep "Results saved to:" | sed 's/Results saved to: //')

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: Could not find experiment directory in watermarking output:"
    echo "Full watermarking script stdout:"
    echo "$WATERMARK_OUTPUT"
    exit 1
fi

echo "Experiment directory: $EXPERIMENT_DIR"

# Run baseline script
# echo "Running baseline script..."
# BASELINE_OUTPUT_FILE="$EXPERIMENT_DIR/text/baseline_output.txt"
# mkdir -p "$(dirname "$BASELINE_OUTPUT_FILE")"
# 
# PYTHON_BASELINE_CMD="python baselines/top_p_standalone.py \
#     --top_p $TOP_P \
#     --model_id \"$MODEL_ID\" \
#     --prompt \"$PROMPT\" \
#     --temperature $TEMPERATURE \
#     --max_tokens $NUM_TOKENS \
#     --output_file \"$BASELINE_OUTPUT_FILE\""
# 
# echo "Executing: $PYTHON_BASELINE_CMD"
# 
# if eval "$PYTHON_BASELINE_CMD"; then
#     echo "Baseline script completed successfully."
# else
#     echo "ERROR: Baseline script failed. Exit code: $?"
#     # Consider adding stderr/stdout capture for baseline too if it becomes problematic
#     exit 1
# fi

echo "Done! Results saved to: $EXPERIMENT_DIR"
echo "Watermarked output: $EXPERIMENT_DIR/text/token_output.txt"
# echo "Baseline output: $BASELINE_OUTPUT_FILE" 