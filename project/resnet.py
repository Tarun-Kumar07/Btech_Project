import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import snntorch as snn
from snntorch import surrogate

from utils import Classifier, DVSGestureDataModule

import torch
import torch.nn as nn
import torch.nn.functional as F

backpass = surrogate.fast_sigmoid(75)
LIF_PARAMS = lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":backpass,"init_hidden":True}

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False),
            snn.Leaky(**lif_params)
            )

def conv1x1(in_channels,out_channels,stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=1,stride=1,bias=False),
            snn.Leaky(**LIF_PARAMS)
            )

class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None,downsample = None):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels,stride),
            conv3x3(mid_channels, in_channels)
        )
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        out = self.conv(x)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.connect_f == 'ADD':
            out += x
        elif self.connect_f == 'AND':
            out *= x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

class SEWBottleNeckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None, downsample=None):
        super(SEWBottleNeckBlock, self).__init__()
        self.connect_f = connect_f
        self.expansion = 4
        self.conv = nn.Sequential(
            conv1x1(in_channels, mid_channels),
            conv3x3(mid_channels, mid_channels),
            conv1x1(mid_channels, mid_channels*self.expansion)
        )
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        out = self.conv(x)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.connect_f == 'ADD':
            out += x
        elif self.connect_f == 'AND':
            out *= x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class SEWNet(nn.Module):
    pass
