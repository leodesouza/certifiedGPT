# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

python _train_adv_img_trans.py  \
    --batch_size 10 \
    --num_samples 10000 \
    --steps 100 \
    --output "/home/swf_developer/storage/attack/minigpt4_adv/" \