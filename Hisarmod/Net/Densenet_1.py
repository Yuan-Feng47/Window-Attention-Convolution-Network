import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(growth_rate)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.avg_pool(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate, num_classes):
        super(DenseNet, self).__init__()

        self.in_channels = 2 * growth_rate
        self.conv1 = nn.Conv1d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.blocks = []
        for i, num_layers in enumerate(block_config):
            self.blocks.append(DenseBlock(self.in_channels, growth_rate, num_layers))
            self.in_channels += growth_rate * num_layers
            if i != len(block_config) - 1:
                self.blocks.append(TransitionLayer(self.in_channels, self.in_channels // 2))
                self.in_channels = self.in_channels // 2
        self.blocks = nn.Sequential(*self.blocks)

        self.bn2 = nn.BatchNorm1d(self.in_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.blocks(out)
        out = F.relu(self.bn2(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 定义不同 DenseNet 变体的层数配置
growth_rate = 32  # 生长率

block_config_121 = (6, 12, 24, 16)
block_config_169 = (6, 12, 32, 32)
block_config_201 = (6, 12, 48, 32)
block_config_264 = (6, 12, 64, 48)


def densenet121(num_classes):
    model = DenseNet(block_config_121, growth_rate, num_classes)
    return model


def densenet169(num_classes):
    model = DenseNet(block_config_169, growth_rate, num_classes)
    return model


def densenet201(num_classes):
    model = DenseNet(block_config_201, growth_rate, num_classes)
    return model


def densenet264(num_classes):
    model = DenseNet(block_config_264, growth_rate, num_classes)
    return model
