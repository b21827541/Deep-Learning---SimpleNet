# SimpleNet Implementation with PyTorch and DDP

## Project Overview
This repository contains my implementation of the [SimpleNet paper](https://arxiv.org/abs/1608.06037) using PyTorch and Distributed Data Parallel (DDP) for efficient multi-GPU training. SimpleNet is a lightweight convolutional neural network architecture that achieves competitive accuracy.

## Hardware & Implementation Details
I utilized a multi-GPU setup for distributed training:
- 2x NVIDIA RTX 3060 Ti
- 1x NVIDIA RTX 3070
- 1x NVIDIA RTX 3080

The implementation leverages PyTorch's Distributed Data Parallel (DDP) framework to efficiently distribute training across multiple GPUs, significantly reducing training time while maintaining model performance.

## Citation
```bibtex
@article{hasanpour2016lets,
  title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
  author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:1608.06037},
  year={2016}
}
