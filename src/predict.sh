#!/usr/bin/env bash
set -e
set -v
# python src/myprogram.py test --work_dir work --test_data $1 --test_output $2
# src/predict.sh data/input.txt data/pred.txt
python src/generate.py --model_dir work/checkpoint-7000 --num_candidates 3 --test_data $1 --test_output $2