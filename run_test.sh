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
TYPE="pipeline"
MAX_ITERATION=60
OUTPUT_DIR="results"

# INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${TYPE}.json"
# EVAL_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${TYPE}_eval_scores.jsonl"

# Alternative output paths (disabled)
INFERENCE_OUTPUT_FILE="results/factual_struct_gpt_base.json"
EVAL_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${TYPE}_structgpt_eval_scores.jsonl"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Decide which dataset name to send to run_inference.py
NAME_FOR_SCRIPT="$DATASET_NAME"
if [[ "$DATASET_NAME" == *"webnlg"* ]]; then
    NAME_FOR_SCRIPT="webnlg_hf"
fi
# echo "Passing --name=$NAME_FOR_SCRIPT"

# ##### Run Python prediction script
# python run_inference.py \
#   --model_provider "$SUPPLIER" \
#   --name "$NAME_FOR_SCRIPT" \
#   --split "$SPLIT" \
#   --type "$TYPE" \
#   --output_file "$INFERENCE_OUTPUT_FILE" \
#   --max_iteration "$MAX_ITERATION"

#### Run evaluation
# python run_evaluation.py \
#   --input_file "$INFERENCE_OUTPUT_FILE" \
#   --dataset_name "$DATASET_NAME" \
#   --dataset_split "$SPLIT" \
#   --output_file "$EVAL_OUTPUT_FILE"


# command="comet-score -s results/triples -t results/webnlg_e2e results/webnlg_agent results/factual_struct_gpt_base -r results/reference0"


# files = ("webnlg_e2e" "webnlg_agent" "factual_struct_gpt_base")
# # Loop through each file and run comet-score
# for result in "${files[@]}"; do
#     command="comet-score -s results/triples -t results/${result} -r ../results/reference0"
#     echo "Running: ${command}"
#     eval $command
# done


# Usage:
# chmod +x run_test.sh
# ./run_test.sh
