# Experiments

This document describes how to replicate our results.

First, train the models on VQAv2:

```
python launch.py train configs/train_configs/vqav2_finetuning_noise_0.yaml
python launch.py train configs/train_configs/vqav2_finetuning_noise_0.25.yaml
python launch.py train configs/train_configs/vqav2_finetuning_noise_0.5.yaml
python launch.py train configs/train_configs/vqav2_finetuning_noise_1.0.yaml
```

Then, certify a subsample of the test set on VQAv2:

```
python launch.py certify configs/train_configs/vqav2_finetuning_noise_0.yaml
python launch.py certify configs/train_configs/vqav2_finetuning_noise_0.25.yaml
python launch.py certify configs/train_configs/vqav2_finetuning_noise_0.5.yaml
python launch.py certify configs/train_configs/vqav2_finetuning_noise_1.0.yaml

```


Prediction experiments on VQAv2:
```
python launch.py predict configs/train_configs/vqav2_finetuning_noise_0.yaml
python launch.py predict configs/train_configs/vqav2_finetuning_noise_0.25.yaml
python launch.py predict configs/train_configs/vqav2_finetuning_noise_0.5.yaml
python launch.py predict configs/train_configs/vqav2_finetuning_noise_1.0.yaml

```

Finally, to visualize noisy images:
```
 ...
 
```