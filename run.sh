#!/bin/bash

# Navigate to the project folder
cd "/mnt/e/pesquisa_ia/projetos/certifiedGPT" || exit 1

# Run the Python script with the specified configuration file
python3 train.py --config-path "/mnt/e/pesquisa_ia/projetos/certifiedGPT/configs/train_configs/cc_sbu_finetuning.yaml"

# Pause equivalent (Optional: keep the terminal open to review output)
echo "Press any key to continue..."
read -n 1 -s
