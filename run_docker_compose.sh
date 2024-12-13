#!/bin/bash
export DATA_DIR=/home/leonardosouza/projects/datasets/vqav2/
export VICUNA_DIR=/home/leonardosouza/projects/checkpoints/vicuna-7b/
export BLIP_FLANT5_PTH=/home/leonardosouza/projects/checkpoints/blip/...pth
export EVA_VIT_G_PTH=/home/leonardosouza/projects/checkpoints/eva_vit_g/..pth


# don't forget to make this script executable
# chmod +x run_docker_compose.sh

docker-compose up