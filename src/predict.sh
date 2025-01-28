#!/usr/bin/env bash
set -e
set -v
# python src/myprogram.py test --work_dir work --test_data $1 --test_output $2
python src/generate.py --model_dir ./char_gpt2_finetune/checkpoint-7201 --start_string "Hello wor" --length 50 --output_file pred.txt