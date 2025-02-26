#!/usr/bin/env bash
set -e
set -v
# python src/myprogram.py test --work_dir work --test_data $1 --test_output $2
# src/predict.sh data/input.txt data/pred.txt
# src/predict.sh data/input_stress.txt data/pred_stress.txt
python src/generate_word.py --model_dir ./checkpoints --top_k 100 --test_input $1 --test_output $2