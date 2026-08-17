import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super().__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, input_batch):
        batch = F.relu(self.bn1a(self.conv1a(input_batch)))
        batch = self.conv2a(batch)
        residual = input_batch if self.stride == 1 else self.downsample(input_batch)
        batch = batch + residual
        intermediate = batch
        batch = F.relu(self.outbna(batch))
        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        return F.relu(self.outbnb(batch + intermediate))


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

    def forward(self, input_batch):
        batch = self.layer4(self.layer3(self.layer2(self.layer1(input_batch))))
        return self.avgpool(batch)


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = ((y - mean) ** 2).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        return self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta


class visualFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet()

    def forward(self, input_batch):
        input_batch = input_batch.transpose(0, 1).transpose(1, 2)
        batchsize = input_batch.shape[0]
        batch = self.frontend3D(input_batch)
        batch = batch.transpose(1, 2)
        batch = batch.reshape(
            batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]
        )
        output = self.resnet(batch)
        output = output.reshape(batchsize, -1, 512)
        output = output.transpose(1, 2)
        return output.transpose(1, 2).transpose(0, 1)


class DSConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, stride=1, padding=1, dilation=1, groups=512, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(512),
            nn.Conv1d(512, 512, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x) + x


class visualTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[DSConv1d() for _ in range(5)])

    def forward(self, x):
        return self.net(x)


class visualConv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
        )

    def forward(self, x):
        return self.net(x)
