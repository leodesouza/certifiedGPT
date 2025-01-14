#!/bin/bash

export PJRT_DEVICE=TPU

NOISE_LEVEL=0.25
MAX_EPOCHS=5
BATCH_SIZE=0
CHECKPOINT_NAME="checkpoint_cc_sbu_align"
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/certifiedgpt_finetune1.yaml'
CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml'

# Execute the Python training script
python3 train.py \
    --config-path=${CONFIG_PATH} \
    --noise_level=${NOISE_LEVEL} \
    --max_epochs=${MAX_EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --checkpoint_name=${CHECKPOINT_NAME}     
    