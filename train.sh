#!/bin/bash

NUM_TPU_CORES=8
NOISE_LEVEL=0.25
MAX_EPOCHS=1
BATCH_SIZE=6
CHECKPOINT_NAME="checkpoint_finetune_vqav2_0.25.pth"

# Execute the Python training script
python3 -m torch_xla.distributed.launch --num_cores=${NUM_TPU_CORES} train.py \
    --num_procs=${NUM_TPU_CORES} \
    --config-path="" \
    --noise_level=${NOISE_LEVEL} \
    --max_epochs=${MAX_EPOCHS} \
    --batch_size=${BATCH_SIZE} \    
    --checkpoint_name=${CHECKPOINT_NAME}     
    