# CertifiedGPT: Adversarial Robustness Evaluation of Vision And Language Models Against Targeted Black Box Attacks

## Overview
This repository contains the code for CertifiedGPT, a pipeline to certify the adversarial robustness of multimodal vision and language model(VLM) against
visual perturbations in input images. 

In this work, we explored the concept of robustness certification in order to make a VLM robust against targeted black-box attacks through the use of a randomized smoothing technique. 

First, we fine-tuned the MiniGPT-4 model on a small subset of VQAv2 and applied Gaussian noise to the input images. We also adapted a smoothed method that encapsulates the original decoder and generates the most probable answer, even with the noise in the input images. 

Finally, we evaluated the model against targeted black-box attacks. Our certified version of MiniGPT-4, when evaluated on a small VQAv2 subset, produced statistically significant results, demonstrating that randomized smoothing is a feasible approach to certifying the robustness of VLMs, especially in scenarios where access to high-performance GPU is limited.

This work is inspired by, and builds upon, concepts from the paper [*Certified Adversarial Robustness via Randomized Smoothing*](https://arxiv.org/pdf/1902.02918) by [*Jeremy Cohen*](https://jmcohen.github.io/), Elan Rosenfeld, and [*Zico Kolter*](https://zicokolter.com/).

CertifiedGPT was Developed by Leonardo Souza under the supervision of [*Pedro Nuno de Souza Moura*](https://ppgi.uniriotec.br/professores-do-ppgi/pedro-moura/) and [*Maíra Athanázio de Cerqueira Gatti*](https://ppgi.uniriotec.br/professores-do-ppgi/mgdebayser/) from the [*The Graduate Program in Computer Science (PPGI)*](https://ppgi.uniriotec.br/) at [*UNIRIO*](https://www.unirio.br/).

## Original vs Noisy Image(σ=0.5)
![Alt text](utils/assets/image_noise.png)

The left panel shows the clean image; the right panel applies additive Gaussian noise with variance σ = 0.5, which is used during finetuning, certification, and smoothed prediction under the randomized smoothing technique. For adversarial attacks, this clean–noisy pair can initialize an iterative image-space optimization that learns a perturbation to alter a model’s output, including VLMs such as MiniGPT-4, for specified tasks. Experiments demonstrate that a VQA task can be attacked in a targeted
manner, forcing the model to produce an attacker specified answer while keeping the perturbation visually subtle.


## Key Contributions
- Certified VLM Framework: We present an adaptation of randomized smoothing certification specifically for VLMs performing VQAv2 tasks.

- Decoder-to-Label Certification: We provide a procedure that maps the MiniGPT-4 decoder’s textual outputs to discrete labels and apply the certification algorithm of
  Cohen et al. [2019] with proper confidence bounds, preserving theoretical guarantees while operating on generated answers.

- Robustness Guarantees: The certified model ensures that small, imperceptible perturbations on input images do not change the predicted answer within an ℓ2 radius
  determined by the smoothing parameters.

- Evaluation Protocol for VQA: We establish an evaluation protocol for VQA under certification that reports
  certified accuracy, coverage, and abstention over the normalized answer set, providing a reproducible setup for future studies.

- Targeted Black-Box Evaluation: We evaluate robustness under targeted black-box adversarial attacks, reporting attack success rate.

- Practical Scalability: We demonstrate a computationally feasible approach that operates with limited GPU resources.


## Methodology
This research is associated with a master's dissertation defended at UNIRIO in 2025. The official version of the dissertation will be made available on the UNIRIO institutional digital library and the Sucupira Platform once the publication process is complete. The corresponding research article has been submitted for peer review but is not yet published.

## Project Status
This research is associated with a master's dissertation defended at UNIRIO in 2025. The official version of the dissertation will be made available on the UNIRIO institutional digital library and the Sucupira Platform once the publication process is complete. The corresponding research article has been submitted for peer review but is not yet published.


## Dataset
https://huggingface.co/LeoSouza/certifiedgpt/tree/main
...
## License
This repository is under [BSD 3-Clause License](Licences/LICENSE.md).
Many codes are based on [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) with 
BSD 3-Clause License [here](Licences/LICENSE_MiniGPT-4.md).

## Project Status
This research is associated with a master's dissertation defended at UNIRIO in 2025. The official version of the dissertation will be made available on the UNIRIO institutional digital library and the Sucupira Platform once the publication process is complete. The corresponding research article has been submitted for peer review but is not yet published.

