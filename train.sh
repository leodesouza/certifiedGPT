#!/bin/bash

export DATA_DIR=/home/leonardosouza/storage/datasets/vqav2

export VICUNA_DIR=/home/leonardosouza/storage/checkpoints/vicuna-7b/

export BLIP_FLANT5_PTH=/home/leonardosouza/storage/checkpoints/blip_flant5/blip2_pretrained_flant5xxl.pth

export EVA_VIT_G_PTH=/home/leonardosouza/storage/checkpoints/eva_vit_g/eva_vit_g.pth

export MINIGPT_4=/home/leonardosouza/storage/checkpoints/minigpt4_stage2/finetuned_minigpt4_7b_stage2_0.pth

export PJRT_DEVICE=TPU

NOISE_LEVEL=0
MAX_EPOCHS=5
BATCH_SIZE=6
CHECKPOINT_NAME="finetuned_certifiedgpt_vqa"
CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/certifiedgpt_finetune1.yaml'
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml'

# Execute the Python training script

python3 train.py \
        --config-path=${CONFIG_PATH} \
        --noise_level=${NOISE_LEVEL} \
        --max_epochs=${MAX_EPOCHS} \
        --batch_size=${BATCH_SIZE} \
        --checkpoint_name=${CHECKPOINT_NAME}
    
# nohup python3 train.py \
#         --config-path=${CONFIG_PATH} \
#         --noise_level=${NOISE_LEVEL} \
#         --max_epochs=${MAX_EPOCHS} \
#         --batch_size=${BATCH_SIZE} \
#         --checkpoint_name=${CHECKPOINT_NAME} > training.log 2>&1 &




#tail -f training.log    #check the progress    

# ps aux | grep train.py    # to find the process
# kill -9 PROCESS_ID        # to kill it if needed
    