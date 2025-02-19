#!/bin/bash

export PJRT_DEVICE=TPU

CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/train_configs/vqav2_finetuning.yaml'

# CHECKPOINT_NAME="finetuned_minigpt4_7b_stage2_0_.pth"
# CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml'

# Execute the Python training script

python3 train.py \
        --config-path=${CONFIG_PATH}        
