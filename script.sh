#!/bin/bash
# Let's call this script venv.sh
# Epsilon was 6/10
#per device reduced from 550 to 128

python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Linear \
  --epsilon 30 \
  --accountant RDP \
  --per_device_train_batch_size 128 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/