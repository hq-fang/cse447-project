#!/usr/bin/env bash
set -e
set -v
# python src/myprogram.py test --work_dir work --test_data $1 --test_output $2
# src/predict.sh data/input.txt data/pred.txt
# src/predict.sh data/input_test.txt data/pred_test.txt
# src/predict.sh data/input_val.txt data/pred_val.txt
python src/generate_word.py --model_dir work/checkpoints-qwen --top_k 500 --test_input $1 --test_output $2