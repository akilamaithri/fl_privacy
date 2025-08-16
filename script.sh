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
# --learning_rate 2e-6\ for QNLI and QST
# weight_decay means the model is regularized to prevent overfitting. 0.01 means a small amount of regularization is applied.
# num_train_epochs 1 means the model is trained for one epoch, which is one complete pass through the training data.

python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Exp \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 128 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \