import torch
import torch.nn as  nn
import torch.nn.functional as F

from utils import Classifier

import snntorch as snn
from snntorch import surrogate

backpass = surrogate.fast_sigmoid(75)
LIF_PARAMS = lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":backpass,"init_hidden":True}

def conv3x3(in_channels,out_channels,stride):
    return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            snn.Leaky(**lif_params)
            )

def conv1x1(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            snn.Leaky(**lif_params)
            )

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = conv1x1(in_channels, out_channels)
        # self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        # self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = conv1x1(out_channels, out_channels*self.expansion)
        # self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        # self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        # x = self.relu(self.batch_norm1(self.conv1(x)))
        
        # x = self.relu(self.batch_norm2(self.conv2(x)))
        
        # x = self.conv3(x)
        # x = self.batch_norm3(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x*=identity
        # x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        # self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        # self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        # self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      # x = self.relu(self.batch_norm2(self.conv1(x)))
      # x = self.batch_norm2(self.conv2(x))
      x = self.conv1(x)
      x = self.conv2(x)

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x *= identity
      # x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.leaky = snn.Leaky(**LIF_PARAMS)
        # self.batch_norm1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = (self.leaky(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                snn.Leaky(**LIF_PARAMS) 
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, channels=2):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=2):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=2):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

def test():
    res = [ResNet50,ResNet101,ResNet152]

    x = torch.rand(size=(16,10,2,128,128))
    for r in res:
        print(r.__name__)
        clf = Classifier(backbone=r(11))
        y = clf(x)
        print(y.shape)
        break

if __name__ == "__main__":
    test()
