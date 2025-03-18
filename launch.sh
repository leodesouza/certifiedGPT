#!/bin/bash

export PJRT_DEVICE=TPU

CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/eval_configs/vqav2_finetuning_noise_0.yaml'

python3 launch.py train --config-path=${CONFIG_PATH}        
