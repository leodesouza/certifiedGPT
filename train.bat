@echo off
cd /d "E:\pesquisa_ia\projetos\certifiedGPT"  # Navigate to the project folder

@REM python train.py --config-path "e:\pesquisa_ia\projetos\certifiedGPT\configs\train_configs\certifiedgpt_finetune1.yaml"
python train.py --config-path "e:\pesquisa_ia\projetos\certifiedGPT\configs\train_configs\cc_sbu_finetuning.yaml"
pause  # Keeps the window open after execution to see any output
