#!/bin/bash

export PJRT_DEVICE=TPU

# CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/certify_configs/vqav2_certify_noise_0.25.yaml'
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/vqav2_finetuning_noise_0.25.yaml'
CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/predict_configs/vqav2_predict_noise_0.25_n_100.yaml'

python3 launch.py smoothing_predict --config-path=${CONFIG_PATH}
