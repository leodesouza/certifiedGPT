#!/bin/bash

export PJRT_DEVICE=TPU

CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/train_configs/vqav2_finetuning_noise_0.5.yaml'

python3 train.py \
        --config-path=${CONFIG_PATH}        
