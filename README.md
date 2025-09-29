# ResNet-18: JAX vs PyTorch Comparison

This repository implements ResNet-18 for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) classification in both **JAX/Flax** and **PyTorch** (run on TPUs), providing a direct framework comparison through identical architectures and datasets.

## **What & Why**

Hands-on exploration of JAX's core concepts ([jit](https://docs.jax.dev/en/latest/jit-compilation.html), [grad](https://docs.jax.dev/en/latest/automatic-differentiation.html), [vmap](https://docs.jax.dev/en/latest/automatic-vectorization.html)) vs PyTorch's approach through the same [ResNet](https://arxiv.org/pdf/1512.03385)-18 model and data.

## **ResNet-18 Architecture**

Following the [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) implementation of `resnet18`:

**Input**: `(N, 3, 32, 32)`

**Stem**
| Operation | Details | Output Shape |
| :--- | :--- | :--- |
| **Conv3x3** | S=1, Out=64 | (N, 64, 32, 32) |
| BN + ReLU | Batch Normalization + Activation | (N, 64, 32, 32) |

**Layer 1-4 (2 BasicBlocks each)**
| Layer | Details | Output Shape |
| :--- | :--- | :--- |
| **Layer 1** | 2×BasicBlock, S=1, Out=64 | (N, 64, 32, 32) |
| **Layer 2** | 2×BasicBlock, S=2→1, Out=128, Downsample | (N, 128, 16, 16) |
| **Layer 3** | 2×BasicBlock, S=2→1, Out=256, Downsample | (N, 256, 8, 8) |
| **Layer 4** | 2×BasicBlock, S=2→1, Out=512, Downsample | (N, 512, 4, 4) |

**Head**
| Operation | Details | Output Shape |
| :--- | :--- | :--- |
| **AvgPool** | Global Average Pooling (4×4→1×1) | (N, 512, 1, 1) |
| **Linear** | 512 → num_classes | (N, 10) |

*BasicBlock*: Two 3×3 Conv layers with BN+ReLU plus identity shortcut connection.

##  **Quick Start**

```bash
# PyTorch TPU
python train_pytorch_tpu.py

# JAX TPU (TODO)
python train_jax_tpu.py
```

*Educational comparison focused on framework differences rather than SOTA performance.*
