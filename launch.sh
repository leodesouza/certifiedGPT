#!/bin/bash

export PJRT_DEVICE=TPU

CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/eval_configs/vqav2_eval_noise_0.yaml'

python3 launch.py eval --config-path=${CONFIG_PATH}        
