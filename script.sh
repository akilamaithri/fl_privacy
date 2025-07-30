#!/bin/bash
# Let's call this script venv.sh
# Epsilon was 6/10
#per device reduced from 550 to 128

#added new
# --num_train_epochs 1 \
# --warmup_steps 0 \
# --weight_decay 0.01 \
# --max_train_samples 1000000 \
#--max_eval_samples 1000000 \

python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Linear \
  --epsilon 31 \
  --accountant RDP \
  --per_device_train_batch_size 128 \
  --num_train_epochs 1 \
  --warmup_steps 0 \
  --weight_decay 0.01 \
  --max_train_samples 1000000 \
  --max_eval_samples 1000000 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/