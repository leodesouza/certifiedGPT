#!/bin/bash

# Set the number of TPU cores
export NUM_TPU_CORES=8

# Execute the Python training script
python3 -m torch_xla.distributed.xla_spawn --num_devices=${NUM_TPU_CORES} train.py \
    --config-path="" \
    --noise_level=0.25 \
    --num_procs=${NUM_TPU_CORES}
