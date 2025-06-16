#!/bin/bash

# Configuration
SUPPLIER="openai"               # e.g., openai, ollama, hf, aixplain
DATASET_NAME="webnlg"
SPLIT="test"
MAX_ITERATION=60
OUTPUT_FILE="results/${DATASET_NAME}.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Python prediction script
python run_test.py \
  --model_provider "$SUPPLIER" \
  --name "$DATASET_NAME" \
  --split "$SPLIT" \
  --output_file "$OUTPUT_FILE" \
  --max_iteration "$MAX_ITERATION"

# chmod +x run_test.sh
# ./run_test.sh