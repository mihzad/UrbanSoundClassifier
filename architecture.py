import torch
from torch import nn as nn, Tensor
from functools import partial
import math


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def GN(num_channels):
    g_approx = math.sqrt(num_channels)
    g_actual = 2 ** round(math.log2(g_approx))  # map to the nearest pow of 2

    while num_channels % g_actual != 0:
        g_actual = g_actual // 2
        if g_actual == 0:
            g_actual = 1
            break

    return nn.GroupNorm(num_groups=g_actual, num_channels=num_channels)


class InvertedResidualConfig:
    def __init__(
        self,
        inp: int,
        kernel: int | tuple[int, int],
        exp: int,
        out: int,
        se: bool,
        activation: str,
        stride: int | tuple[int, int],
        width_mult: float = 1.0,
        expand: bool = True
    ):
        self.input_channels = self.adjust_channels(inp, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(exp, width_mult)
        self.out_channels = self.adjust_channels(out, width_mult)
        self.use_se = se
        self.use_hs = (activation == "HS") #True => HS; False => LReLU
        self.stride = stride
        self.expand = not (expand is False and inp == exp) #can`t set expand = False for inp != exp

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels*width_mult, divisible_by=8)



class SqueezeExcite2D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcite2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #N,C,1,1
        self.squeeze_channels = make_divisible(in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.squeeze_channels, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.squeeze_channels, in_channels, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale.expand_as(x)

class InvertedResidual2D(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: nn.BatchNorm2d = None):
        super().__init__()

        self.cnf = cnf
        self.norm_layer = norm_layer
        if self.norm_layer is None:
            self.norm_layer = partial(nn.BatchNorm2d, eps=1e-8, momentum=0.01)

        self.use_residual_connection = (cnf.input_channels == cnf.out_channels and cnf.stride == 1)
        layers: list[nn.Module] = []

        if cnf.use_hs:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)

        if cnf.expand:
            # expand
            layers.append(nn.Conv2d(
                in_channels=cnf.input_channels,
                out_channels=cnf.expanded_channels,
                kernel_size=1,
                stride=1, padding=0, bias=False))
            layers.append(self.norm_layer(cnf.expanded_channels))
            layers.append(self.activation)

        # depth-wise
        layers.append(nn.Conv2d(
            in_channels=cnf.expanded_channels,
            out_channels=cnf.expanded_channels,
            kernel_size=cnf.kernel,
            groups=cnf.expanded_channels,
            stride=cnf.stride,
            padding=( (cnf.kernel-1) // 2,
                      (cnf.kernel-1) // 2
                      ),
            bias=False))
        layers.append(self.norm_layer(cnf.expanded_channels))
        layers.append(self.activation)

        if cnf.use_se:
            layers.append(SqueezeExcite2D(in_channels=cnf.expanded_channels))

        # project
        layers.append(nn.Conv2d(
            in_channels=cnf.expanded_channels,
            out_channels=cnf.out_channels,
            kernel_size=1,
            stride=1, padding=0, bias=False))
        layers.append(self.norm_layer(cnf.out_channels))
        # no activation

        # converting created layers list
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual_connection:
            return x + self.block(x)
        return self.block(x)


class MobileNetAudio(nn.Module):
    def __init__(self, n_classes=10, width_mult=1):
        super().__init__()

        self.norm_layer = partial(nn.BatchNorm2d, eps=1e-8, momentum=0.01)

        # define InvertedResidual blocks
        self.configs: list[InvertedResidualConfig] = [
            InvertedResidualConfig(inp=16, exp=16, out=16, kernel=3, stride=2,
                                   se=True, activation='RE', width_mult=width_mult, expand=False),
            InvertedResidualConfig(inp=16, exp=72, out=24, kernel=3, stride=2,
                                   se=False, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=24, exp=88, out=24, kernel=3, stride=1,
                                   se=False, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=24, exp=96, out=40, kernel=5, stride=2,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=40, exp=240, out=40, kernel=5, stride=1,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=40, exp=240, out=40, kernel=5, stride=1,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=40, exp=120, out=48, kernel=5, stride=1,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=48, exp=144, out=48, kernel=5, stride=1,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=48, exp=288, out=96, kernel=5, stride=2,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=96, exp=576, out=96, kernel=5, stride=1,
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=96, exp=576, out=96, kernel=5, stride=1,
                                   se=True, activation='HS', width_mult=width_mult),
        ]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.configs[0].input_channels,
                kernel_size=3,
                stride=2, bias=False),
            self.norm_layer(self.configs[0].input_channels),
            nn.Hardswish()
        )

        self.blocks = nn.ModuleList()
        for cfg in self.configs:
                self.blocks.append(InvertedResidual2D(cnf=cfg))

        # Final layers
        classifier_fc1_in_features = max(make_divisible(576*width_mult), 128)
        classifier_fc2_in_features = max(make_divisible(1024*width_mult), 128)
        self.final_conv = nn.Sequential(
            nn.Conv2d(make_divisible(self.configs[-1].out_channels), classifier_fc1_in_features,
                      kernel_size=1, padding=0, bias=False),
            self.norm_layer(classifier_fc1_in_features),
            nn.Hardswish(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=classifier_fc1_in_features,
                      out_features=classifier_fc2_in_features, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.4),
            nn.Linear(in_features=classifier_fc2_in_features, out_features=n_classes, bias=True)
        )

    def forward(self, x: Tensor):

        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x
