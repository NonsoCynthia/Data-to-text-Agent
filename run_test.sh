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
TYPE="agent"
MAX_ITERATION=60
OUTPUT_DIR="results"

INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${TYPE}.json"
EVAL_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${TYPE}_eval_scores.jsonl"

# INFERENCE_OUTPUT_FILE="/home/chinonso/Documents/output_agent3/webnlg_agent.json"
# EVAL_OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_${TYPE}_output_agent3_scores.jsonl"

### Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Decide which dataset name to send to run_inference.py
NAME_FOR_SCRIPT="$DATASET_NAME"
if [[ "$DATASET_NAME" == *"webnlg"* ]]; then
    NAME_FOR_SCRIPT="webnlg_hf"
fi
# echo "Passing --name=$NAME_FOR_SCRIPT"

#### Run Python prediction script
python run_inference.py \
  --model_provider "$SUPPLIER" \
  --name "$NAME_FOR_SCRIPT" \
  --split "$SPLIT" \
  --type "$TYPE" \
  --output_file "$INFERENCE_OUTPUT_FILE" \
  --max_iteration "$MAX_ITERATION"


# #### === Step 2: Run evaluation ===
# python run_evaluation.py \
#   --input_file "$INFERENCE_OUTPUT_FILE" \
#   --dataset_name "$DATASET_NAME" \
#   --dataset_split "$SPLIT" \
#   --output_file "$EVAL_OUTPUT_FILE"


### Usage:
### chmod +x run_test.sh
### ./run_test.sh
