#!/bin/bash

export PJRT_DEVICE=TPU

# CHECKPOINT_NAME="vqav2_finetuning.yaml"
# CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/train_configs/certifiedgpt_finetune1.yaml'

CHECKPOINT_NAME="finetuned_minigpt4_7b_stage2_0_.pth"
CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml'

# Execute the Python training script

python3 train.py \
        --config-path=${CONFIG_PATH} \
        --checkpoint_name=${CHECKPOINT_NAME}
