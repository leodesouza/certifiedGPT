# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

python _train_adv_img_trans.py  \
    --batch_size 1 \
    --num_samples 1 \
    --steps 1 \
    --output "/home/swf_developer/storage/attack/minigpt4_adv/" \