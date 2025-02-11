#!/bin/bash

export PJRT_DEVICE=TPU

NOISE_LEVEL=0
MAX_EPOCHS=1
BATCH_SIZE=32
CHECKPOINT_NAME="finetuned_certifiedgpt_vqa"
CONFIG_PATH='/home/swf_developer/certifiedGPT/configs/train_configs/certifiedgpt_finetune1.yaml'
# CONFIG_PATH='/home/leonardosouza/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml'

# Execute the Python training script

python3 train.py \
        --config-path=${CONFIG_PATH} \
        --noise_level=${NOISE_LEVEL} \
        --max_epochs=${MAX_EPOCHS} \
        --batch_size=${BATCH_SIZE} \
        --checkpoint_name=${CHECKPOINT_NAME} 2>&1 | tee training_report.log
    
# nohup python3 train.py \
#         --config-path=${CONFIG_PATH} \
#         --noise_level=${NOISE_LEVEL} \
#         --max_epochs=${MAX_EPOCHS} \
#         --batch_size=${BATCH_SIZE} \
#         --checkpoint_name=${CHECKPOINT_NAME} > training.log 2>&1 &




#tail -f training.log    #check the progress    

# ps aux | grep train.py    # to find the process
# kill -9 PROCESS_ID        # to kill it if needed
    