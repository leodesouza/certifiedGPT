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
python launch.py certify configs/train_configs/vqav2_finetuning_noise_0.yaml
python launch.py certify configs/train_configs/vqav2_finetuning_noise_0.25.yaml
python launch.py certify configs/train_configs/vqav2_finetuning_noise_0.5.yaml
python launch.py certify configs/train_configs/vqav2_finetuning_noise_1.0.yaml

```

Finally, to visualize noisy images:
```
python code/visualize.py imagenet figures/example_images/imagenet 100 0.0 0.25 0.5 1.0
python code/visualize.py imagenet figures/example_images/imagenet 5400 0.0 0.25 0.5 1.0
python code/visualize.py imagenet figures/example_images/imagenet 9025 0.0 0.25 0.5 1.0
python code/visualize.py imagenet figures/example_images/imagenet 19411 0.0 0.25 0.5 1.0

python code/visualize.py cifar10 figures/example_images/cifar10 10 0.0 0.25 0.5 1.0
python code/visualize.py cifar10 figures/example_images/cifar10 20 0.0 0.25 0.5 1.0
python code/visualize.py cifar10 figures/example_images/cifar10 70 0.0 0.25 0.5 1.0
python code/visualize.py cifar10 figures/example_images/cifar10 110 0.0 0.25 0.5 1.0
```