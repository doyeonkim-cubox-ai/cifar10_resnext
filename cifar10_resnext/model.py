import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def pick(m):
    if m == 'wresnet':
        return wresnet()
    elif m == 'resnext29_8':
        return resnext29_8()
    elif m == 'resnext29_16':
        return resnext29_16()
    else:
        print(f'No such model exists: {m}')
        exit(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, opt=0, cardinality=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes*cardinality, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*cardinality)
        self.conv2 = nn.Conv2d(planes*cardinality, planes*cardinality*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*cardinality*self.expansion)
        self.relu = nn.ReLU()

        self.stride = stride
        self.opt = opt
        self.downsample = None
        if opt != 0:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes*cardinality*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
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


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, opt=0, cardinality=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes*cardinality, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*cardinality)
        self.conv2 = nn.Conv2d(planes*cardinality, planes*cardinality, kernel_size=3, groups=cardinality, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*cardinality)
        self.conv3 = nn.Conv2d(planes*cardinality, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride
        self.opt = opt
        self.downsample = None
        if opt != 0:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=1, num_classes=10):
        super(ResNeXt, self).__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layers(block, layers[0], 64, stride=1, cardinality=cardinality)
        self.layer2 = self._make_layers(block, layers[1], 128, stride=2, cardinality=cardinality)
        self.layer3 = self._make_layers(block, layers[2], 256, stride=2, cardinality=cardinality)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _make_layers(self, block, blocks, planes, stride, cardinality):
        opt = 0
        if stride != 1 or self.in_planes != planes*block.expansion:
            opt = self.in_planes
        layers = []
        layers.append(block(self.in_planes, planes, stride, opt, cardinality))
        self.in_planes = planes*block.expansion
        for i in (1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, cardinality=cardinality))

        return nn.Sequential(*layers)


class WideResNet(nn.Module):
    # Just for convenience use width as cardinality here
    def __init__(self, block, layers, cardinality=1, num_classes=10):
        super(WideResNet, self).__init__()

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layers(block, layers[0], 16*cardinality, stride=1, cardinality=cardinality)
        self.layer2 = self._make_layers(block, layers[1], 32*cardinality, stride=2, cardinality=cardinality)
        self.layer3 = self._make_layers(block, layers[2], 64*cardinality, stride=2, cardinality=cardinality)

        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(640, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _make_layers(self, block, blocks, planes, stride, cardinality):
        opt = 0
        if stride != 1 or self.in_planes != planes*block.expansion:
            opt = self.in_planes
        layers = []
        layers.append(block(self.in_planes, planes, stride, opt))
        self.in_planes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))

        return nn.Sequential(*layers)


def wresnet():
    return WideResNet(BasicBlock, [4, 4, 4], cardinality=10)


def resnext29_8():
    return ResNeXt(BottleNeck, [3, 3, 3], cardinality=8)


def resnext29_16():
    return ResNeXt(BottleNeck, [3, 3, 3], cardinality=16)
