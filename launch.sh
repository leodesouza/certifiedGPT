#!/bin/bash

export PJRT_DEVICE=TPU

CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/certify_configs/vqav2_certify_noise_0.25.yaml'

python3 launch.py train --config-path=${CONFIG_PATH}        
