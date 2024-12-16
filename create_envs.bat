@echo off

REM Set the DATA_DIR environment variable
set DATA_DIR=E:\pesquisa_ia\projetos\datasets\vqav2\
echo DATA_DIR set to "%DATA_DIR%"

REM Set the VICUNA_DIR environment variable
set VICUNA_DIR=E:\pesquisa_ia\projetos\checkpoints\vicuna-7b\
echo VICUNA_DIR set to "%VICUNA_DIR%"

REM Set the BLIP_FLANT5_PTH environment variable
set BLIP_FLANT5_PTH=E:\pesquisa_ia\projetos\checkpoints\blip_flant5\blip2_pretrained_flant5xxl.pth
echo BLIP_FLANT5_PTH set to "%BLIP_FLANT5_PTH%"

REM Set the EVA_VIT_G_PTH environment variable
set EVA_VIT_G_PTH=E:\pesquisa_ia\projetos\checkpoints\eva_vit_g\eva_vit_g.pth
echo EVA_VIT_G_PTH set to "%EVA_VIT_G_PTH%"

echo Environment variables have been set successfully for this session.
pause
