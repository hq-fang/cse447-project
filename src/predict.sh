#!/usr/bin/env bash
set -e
set -v
# python src/myprogram.py test --work_dir work --test_data $1 --test_output $2
# src/predict.sh data/input.txt data/pred.txt
# src/predict.sh data/input_test.txt data/pred_test.txt
# src/predict.sh data/input_val.txt data/pred_val.txt
# src/predict.sh data/input_val_eng.txt data/pred_val_eng.txt
# src/predict.sh data/input_val_eng_short.txt data/pred_val_eng_short.txt
# src/predict.sh data/input_val_eng_long.txt data/pred_val_eng_long.txt
# src/predict.sh data/input_val_russian.txt data/pred_val_russian.txt
python src/generate_word_plus_char.py --model_dir work/checkpoints-llama --top_k 100 --threshold 0.0 --test_input $1 --test_output $2