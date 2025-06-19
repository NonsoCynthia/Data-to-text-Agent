#!/bin/bash

# Activate Conda environment
# For macOS (commented out):
# source /Users/chinonsoosuji/opt/anaconda3/etc/profile.d/conda.sh

# For Ubuntu/Linux:
source /home/chinonso/anaconda3/etc/profile.d/conda.sh
conda activate lang2

# Configuration
SUPPLIER="openai"
DATASET_NAME="webnlg"
SPLIT="test"
MAX_ITERATION=60
OUTPUT_DIR="results"
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}.json"
EVAL_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_eval_scores.jsonl"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run Python prediction script
# python run_inference.py \
#   --model_provider "$SUPPLIER" \
#   --name "$DATASET_NAME" \
#   --split "$SPLIT" \
#   --output_file "$INFERENCE_OUTPUT_FILE" \
#   --max_iteration "$MAX_ITERATION"


# === Step 2: Run evaluation ===
# Evaluate predictions and store per-record scores
python run_evaluation.py --input_file "$INFERENCE_OUTPUT_FILE" --output_file "$EVAL_OUTPUT_FILE"


# Usage:
# chmod +x run_test.sh
# ./run_test.sh
