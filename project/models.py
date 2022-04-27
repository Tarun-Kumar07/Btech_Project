from copy import deepcopy
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import snntorch as snn
from snntorch import surrogate, utils

from utils import Classifier, DVSGestureDataModule, quantize

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

def get_model(num_classes):
    conv1 = nn.Conv2d(2,4,5,2,2)
    conv2 = nn.Conv2d(4,8,5,2,2)
    conv3 = nn.Conv2d(8,8,3,2,1)
    conv4 = nn.Conv2d(8,16,3,2,1)
    dropout = nn.Dropout2d()   
    linear = nn.Linear(1024,num_classes)   

    lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75),"init_hidden":True}

    model = nn.Sequential(
                          conv1, 
                          snn.Leaky(**lif_params),
                          conv2,
                          snn.Leaky(**lif_params),
                          conv3,
                          snn.Leaky(**lif_params),
                          conv4,
                          snn.Leaky(**lif_params),
                          dropout,
                          nn.Flatten(),
                          linear,
                          snn.Leaky(**lif_params),
                        )

    return model

class SCNN(nn.Module):

    def __init__(self,lif_params,in_channels,out_channels,kernel,stride=1,bias=True,*args,**kwargs):
        super(SCNN,self).__init__(*args,**kwargs)

        # Need to add padding to not change the x,y dimensions
        need_padding = in_channels == out_channels and stride == 1
        padding = kernel // 2 if need_padding else 0

        conv = nn.Conv2d(in_channels,out_channels,kernel,stride,padding,bias=bias)
        leaky = snn.Leaky(**lif_params)

        self.model = nn.Sequential(conv,leaky)
    
    def forward(self,x):
        # print(x)
        # x = self.conv(x)
        # # print(x)
        # spk = self.leaky(x)
        # print("==================")

        return self.model(x)

def test_scnn():
    lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75),"init_hidden":True}
    scnn = SCNN(2,4,5,**lif_params)
    input_ = torch.rand((1,2,10,10))
    out_ = scnn(input_)
    print(out_.shape)

class SLINEAR(nn.Module):

    def __init__(self,lif_params,in_features,out_features,bias=True,*args,**kwargs):
        super(SLINEAR,self).__init__(*args,**kwargs)
        linear = nn.Linear(in_features,out_features,bias)
        leaky = snn.Leaky(**lif_params)
        self.model = nn.Sequential(linear,leaky)

    def forward(self,x):
        # x = self.linear(x)
        # x = self.leaky(x)

        return self.model(x)


class CustomSNN(nn.Module):
    def __init__(self,conv_config:list,linear_config:list,in_channels:int,depth:float,width:float,lif_params:dict):
        super(CustomSNN,self).__init__()
        self._in_channels = in_channels
        self.depth_factor = depth 
        self.width_factor = width

        self.lif_params  = lif_params 

        conv_layers = self._create_conv(conv_config)
        linear_layers  = self._create_linear(linear_config)

        model = nn.Sequential(*conv_layers,nn.Flatten(),*linear_layers)
        # self._modules = deepcopy(model._modules)
        self.model = model


    def _create_conv_blocks(self,conv_params) -> nn.Module:
        kernel,out_channels,stride,repeat = conv_params

        repeat = int(repeat * self.depth_factor)
        out_channels = int(out_channels * self.width_factor)

        conv = SCNN(self.lif_params,self._in_channels,out_channels,kernel,stride)
        layers = [conv]
        
        self._in_channels = out_channels

        for _ in range(repeat - 1):
            layers.append(SCNN(self.lif_params,out_channels,out_channels,kernel))
       
        return nn.Sequential(* layers)
    
    def _create_conv(self,conv_config:list)->nn.Module:
        layers = []
        for conv_params in conv_config:
            layers.append(self._create_conv_blocks(conv_params))
        return nn.Sequential(* layers)

    def _create_linear(self,linear_config:list) -> nn.Module:
        layers = []
        for in_features,out_features in linear_config:
            layers.append(SLINEAR(self.lif_params,in_features,out_features))

        return nn.Sequential(* layers)
    

    def forward(self,x):
       return self.model(x) 


def cal_linear_features(in_features,conv_config):
    out_ = 0
    in_ = in_features
    channels = 0
    for kernel,out_channel,stride,repeat in conv_config:
        out_ = int((in_-kernel)/stride) + 1
        in_ = out_
        channels = out_channel
       
    return out_* out_* channels

def test():
    conv_baseline = [ 
                #kernel, out_channels , stride, repeat 
                [5, 4 , 2 , 1],
                [5, 8 , 2 , 2],
                [3, 8 , 2 , 2],
                [3, 16 , 2 , 1],
            ]

    linear_features = cal_linear_features(128,conv_baseline)
    
    linear_baseline = [
                #in_features, out_features
                [linear_features, 11],
            ]

    lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75)}

    params = {
            "conv_config": conv_baseline,
            "linear_config": linear_baseline,
            "in_channels":2,
            "depth":1.0,
            "width":1.0,
            "lif_params":lif_params
        }

    
    model = CustomSNN(**params)
    clf = Classifier(model)
    # print(model)
    # logger = TensorBoardLogger("./logs","test_graph",log_graph=True)
    x = torch.rand(16,150,2,128,128)
    y = clf(x)
    # logger.log_graph(clf,input_array = x)
    # logger = TensorBoardLogger("./logs",name="test_class_1",log_graph=True)
    # # model_fxp = quantize(model)
    # clf = Classifier(model)
    # dm = DVSGestureDataModule("./data") 
    # trainer = pl.Trainer(logger = logger,max_epochs=2,fast_dev_run=True)
    # trainer.fit(clf,dm)
    # trainer.test(clf,dm)


def test_dm():
    dm = DVSGestureDataModule("./data") 
    dm.setup()

    train_dl = dm.train_dataloader()

    for x,y in train_dl:
        print(x[0][0])
        break
    


def cli_main():
    # model = Classifier(get_model(11))
    dm = DVSGestureDataModule("./data") 

    # logger = TensorBoardLogger("./logs",name="custom_model")
    # trainer = pl.Trainer(logger=logger,max_epochs=1,fast_dev_run=False)
    # trainer.fit(model,dm)

    # PATH = "./logs/custom_model/version_4/checkpoints/epoch=399-step=6399.ckpt"
    # model = Classifier.load_from_checkpoint(checkpoint_path=PATH,backbone=get_model(11))

    # dm = DVSGestureDataModule("./data") 

    # trainer = pl.Trainer(max_epochs=1,fast_dev_run=False)
    # trainer.test(model,dm)

    # print("Testing quantized")
    # model_fxp = quantize(model.backbone)
    # clf = Classifier(model_fxp)
    # trainer.test(clf,dm)

    # classfier = Classifier.load_from_checkpoint("./logs/test_class/version_0/checkpoints/epoch=1-step=121.ckpt")

    # trainer.test(classfier,dm)
    

if __name__ == "__main__":
    # cli_main()
    test()

