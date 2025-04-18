#!/bin/bash

export PJRT_DEVICE=TPU

# CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/certify_configs/vqav2_certify_noise_0.25.yaml'
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/vqav2_finetuning_noise_0.25.yaml'
CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/eval_configs/vqav2_eval_noise_1.0.yaml'


python3 launch.py eval --config-path=${CONFIG_PATH}
