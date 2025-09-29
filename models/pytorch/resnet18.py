import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List

class BasicBlock(nn.Module):
    expansion: int = 1 # Used for deeper models (ResNet-50), always 1 for ResNet-18

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 1. First Conv, BN, ReLU
        self.conv1 = nn.Conv2d(inplanes, 
                               planes, 
                               kernel_size=3, 
                               stride=stride, 
                               padding=1, 
                               bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # 2. Second Conv, BN
        self.conv2 = nn.Conv2d(planes, 
                               planes, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               bias=False)
        self.bn2 = norm_layer(planes)
        
        # 3. Downsample module
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 64

        # CIFAR-10 Specific:
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity() # no maxpool for cifar10
        
        print("Initing Layer 1...")
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        print("Initing Layer 2...")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        print("Initing Layer 3...")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        print("Initing Layer 4...")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, 
                          planes * block.expansion, 
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        print(f' - BasicBlock 1, inplanes={self.in_planes}, planes={planes}, stride={stride}, downsample={downsample}, norm_layer={norm_layer}')
        layers.append(block(self.in_planes, planes, stride, downsample, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            print(f' - BasicBlock 2, inplanes={self.in_planes}, planes={planes}, norm_layer={norm_layer}')
            layers.append(block(self.in_planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) # 64
        x = self.layer2(x) # 128
        x = self.layer3(x) # 256
        x = self.layer4(x) # 512
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)