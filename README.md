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

IMAGNET 1K: `python imagenet.py <path/to/imagenet_train> <path/to/imagenet_val> <path/to/imagenet_r> <path/to/imagenet_c> <path/to/mixing_set> 1000`

## Mixing set

The mixing set of fractals and feature visualizations used in the paper can be downloaded
[here](some link).

## Pretrained weights

Weights for a ResNet-50 ImageNet classifier trained with PixMix for 90 and 180 epochs are available
[here](some link).
