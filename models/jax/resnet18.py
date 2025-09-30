"""
JAX/Flax ResNet18 model for CIFAR-10
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence
from functools import partial


class BasicBlock(nn.Module):
    """ResNet BasicBlock for ResNet-18/34 architectures."""
    channels: int
    stride: int = 1
    use_bn: bool = True
    norm: Callable = nn.BatchNorm  
    act: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x 
        
        # Main Path
        y = nn.Conv(
            features=self.channels, 
            kernel_size=(3, 3), 
            strides=(self.stride, self.stride),
            padding='SAME',
            use_bias=False,
            name='conv1'
        )(x)
        
        if self.use_bn:
            y = self.norm(use_running_average=not train, name='bn1')(y)
        y = self.act(y)
        
        y = nn.Conv(
            features=self.channels, 
            kernel_size=(3, 3), 
            strides=(1, 1),
            padding='SAME',
            use_bias=False,
            name='conv2'
        )(y)
        
        if self.use_bn:
            y = self.norm(use_running_average=not train, name='bn2')(y)
        
        # Skip Connection 
        if residual.shape[-1] != self.channels or self.stride != 1:
            residual = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                name='projection'
            )(residual)
            
            if self.use_bn:
                residual = self.norm(use_running_average=not train, name='bn_proj')(residual)
        
        # Final Addition and Activation
        out = y + residual
        return self.act(out)


class ResNet(nn.Module):
    """ResNet-18 implementation for CIFAR-10"""
    num_classes: int = 10
    num_blocks: Sequence[int] = (2, 2, 2, 2) 
    
    norm: Callable = partial(nn.BatchNorm, 
                             use_running_average=False, 
                             momentum=0.1, 
                             epsilon=1e-5)
    act: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        def make_stage(input_channels, num_blocks, stride):
            layers = []
            layers.append(
                BasicBlock(channels=input_channels, stride=stride, norm=self.norm, act=self.act)
            )
            for _ in range(1, num_blocks):
                layers.append(
                    BasicBlock(channels=input_channels, stride=1, norm=self.norm, act=self.act)
                )
            return layers

        # CIFAR-10 Stem (no max pooling)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, name='conv_init')(x)
        x = self.norm(use_running_average=not train, name='bn_init')(x)
        x = self.act(x)
        
        # Residual Stages
        x = nn.Sequential(make_stage(64, self.num_blocks[0], stride=1), name='stage1')(x)
        x = nn.Sequential(make_stage(128, self.num_blocks[1], stride=2), name='stage2')(x)
        x = nn.Sequential(make_stage(256, self.num_blocks[2], stride=2), name='stage3')(x)
        x = nn.Sequential(make_stage(512, self.num_blocks[3], stride=2), name='stage4')(x)
        
        # Final Head
        x = jnp.mean(x, axis=(1, 2))  # Global Average Pooling
        x = nn.Dense(features=self.num_classes, name='dense_final')(x)
        
        return x


def resnet18(num_classes: int = 10) -> ResNet:
    """Create ResNet-18 model"""
    return ResNet(num_classes=num_classes)