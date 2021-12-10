# PixMix

<img align="center" src="assets/pixmix.png" width="750">

## Introduction

In real-world applications of machine learning, reliable and safe systems must consider
measures of performance beyond standard test set accuracy. These other goals
include out-of-distribution (OOD) robustness, prediction consistency, resilience to
adversaries, calibrated uncertainty estimates, and the ability to detect anomalous
inputs. However, improving performance towards these goals is often a balancing
act that today’s methods cannot achieve without sacrificing performance on other
safety axes. For instance, adversarial training improves adversarial robustness
but sharply degrades other classifier performance metrics. Similarly, strong data
augmentation and regularization techniques often improve OOD robustness but
harm anomaly detection, raising the question of whether a Pareto improvement on
all existing safety measures is possible. To meet this challenge, we design a new
data augmentation strategy utilizing the natural structural complexity of pictures
such as fractals, which outperforms numerous baselines, is near Pareto-optimal,
and comprehensively improves safety measures.

Read the paper [here](https://arxiv.org/pdf/2112.05135.pdf).

## Pseudocode

<img align="center" src="assets/pixmix_code.png" width="750">

## Contents

`pixmix_utils.py` includes reference implementation of augmentations and mixings used in PixMix.

We also include PyTorch implementations of PixMix on both CIFAR-10/100 and
ImageNet in `cifar.py` and `imagenet.py` respectively, which both support
training and evaluation on CIFAR-10/100-C and ImageNet-C.

## Usage

Training recipes used in our paper:

CIFAR: `python cifar.py <cifar10 or cifar100> <path/to/mixing_set>`

ImageNet 1K: `python imagenet.py <path/to/imagenet_train> <path/to/imagenet_val> <path/to/imagenet_r> <path/to/imagenet_c> <path/to/mixing_set> 1000`

## Mixing Set

The mixing set of fractals and feature visualizations used in the paper can be downloaded
[here](https://drive.google.com/file/d/1qC2gIUx9ARU7zhgI4IwGD3YcFhm8J4cA/view?usp=sharing).

## Pretrained Models
Weights for a 40x4-WRN CIFAR-10/100 classifier trained with PixMix for 100 epochs are available
[here](https://drive.google.com/drive/folders/1tHu2MBU3P9lvgtc06_VaC6AsMqwyYFSA?usp=sharing).

Weights for a ResNet-50 ImageNet classifier trained with PixMix for 90 and 180 epochs are available
[here](https://drive.google.com/drive/folders/1tHu2MBU3P9lvgtc06_VaC6AsMqwyYFSA?usp=sharing).

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2022robustness,
      title={PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures},
      author={Dan Hendrycks and Andy Zou and Mantas Mazeika and Leonard Tang and Dawn Song and Jacob Steinhardt},
      journal={arXiv preprint arXiv:2112.05135},
      year={2022}
    }
