# SimpleNet Implementation with PyTorch and DDP

## Project Overview
This repository contains my implementation of the [SimpleNet paper](https://arxiv.org/abs/1608.06037) using PyTorch and Distributed Data Parallel (DDP) for efficient multi-GPU training. SimpleNet is a lightweight convolutional neural network architecture for image anomaly detection and localization.

## Hardware & Implementation Details
I utilized a multi-GPU setup for distributed training:
- 2x NVIDIA RTX 3060 Ti
- 1x NVIDIA RTX 3070
- 1x NVIDIA RTX 3080

The implementation leverages PyTorch's Distributed Data Parallel (DDP) framework to efficiently distribute training across multiple GPUs, significantly reducing training time while maintaining model performance.

## Citation
```bibtex
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yuansheng and Wang, Zilei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20402--20411},
  year={2023}
}
