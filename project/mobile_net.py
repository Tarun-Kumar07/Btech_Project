import torch
import torch.nn as nn

import snntorch as snn
from snntorch import utils, surrogate

class BottleNeck(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, stride:int, expand_ratio:int,output:bool=False):
        super().__init__()
        backpass = surrogate.fast_sigmoid(75)
        lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":backpass,"init_hidden":True}

        self.residual_connection = False #(stride == 1 and in_channel == out_channel)
        self.layers = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(in_channel * expand_ratio),
                # nn.ReLU6(inplace=True),
                snn.Leaky(**lif_params),

                nn.Conv2d(in_channel * expand_ratio, in_channel * expand_ratio, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel * expand_ratio),
                # nn.BatchNorm2d(in_channel * expand_ratio),
                # nn.ReLU6(inplace=True),
                snn.Leaky(**lif_params),

                nn.Conv2d(in_channel * expand_ratio, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(out_channel)
                snn.Leaky(**lif_params,output=output),
        )
        
    def forward(self, input):
        utils.reset(self.layers)
        x = self.layers(input)
        if self.residual_connection:
            out = x + input
        else:
            out = x
        return out

class SpikingMobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
                )
    def forward(self,input):
        pass  
