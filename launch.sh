#!/bin/bash

#export PJRT_DEVICE=TPU
CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/attack_configs/vqav2_eval_noise_0.yaml'
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/vqav2_finetuning_noise_0.25.yaml'
# CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/predict_configs/vqav2_predict_noise_0.25_n_100.yaml'

# python3 launch.py smoothing_predict --config-path=${CONFIG_PATH}
# python3 launch.py transfer_based_attack --config-path=${CONFIG_PATH}
python3 launch.py img_t2_text --config-path=${CONFIG_PATH}
#python3 launch.py query_based_attack --config-path=${CONFIG_PATH}
# python3 launch.py chat --config-path=${CONFIG_PATH}

