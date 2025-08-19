#!/bin/bash

# Activate Conda environment
source /home/chinonso/anaconda3/etc/profile.d/conda.sh
conda activate webnlgEval

# English
python3 eval.py -R data/en/references/reference -H data/en/factual_struct_gpt_base -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt >> eval_results_factual_struct_gpt_base.txt 2>&1
# #python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt

# # Russian
# python3 eval.py -R data/ru/reference -H data/ru/hypothesis -lng ru -nr 1 -m bleu,meteor,chrf++,ter,bert