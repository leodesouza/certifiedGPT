#!/bin/bash

NUM_TPU_CORES=8
NOISE_LEVEL=0
MAX_EPOCHS=1
BATCH_SIZE=12
CHECKPOINT_NAME="checkpoint_cc_sbu_align.pth"
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/certifiedgpt_finetune1.yaml'
CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml'



# Execute the Python training script
python3 train.py \
    --num_procs=${NUM_TPU_CORES} \
    --config-path=${CONFIG_PATH} \
    --noise_level=${NOISE_LEVEL} \
    --max_epochs=${MAX_EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --checkpoint_name=${CHECKPOINT_NAME}     
    