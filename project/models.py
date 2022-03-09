import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import snntorch as snn
from snntorch import surrogate

from utils import Classifier, DVSGestureDataModule

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

def get_model(num_classes):
    conv1 = nn.Conv2d(2,16,5,2)
    conv2 = nn.Conv2d(16,32,5,2)
    conv3 = nn.Conv2d(32,64,3,2)
    conv4 = nn.Conv2d(64,128,3,2)
    dropout = nn.Dropout2d()
    linear1 = nn.Linear(64*128,512)   
    linear2 = nn.Linear(512,num_classes)   
    backpass = surrogate.fast_sigmoid(75)
    lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":backpass,"init_hidden":True}

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
                          linear1,
                          snn.Leaky(**lif_params),
                          linear2,
                          snn.Leaky(**lif_params,output=True),
                        )

    return model

def cli_main():
    model = Classifier(get_model(11))
    dm = DVSGestureDataModule("./data") 

    logger = TensorBoardLogger("./logs",name="custom_model")
    trainer = pl.Trainer(logger=logger,max_epochs=1,fast_dev_run=False)
    trainer.fit(model,dm)

    trainer.test(model,dm)

if __name__ == "__main__":
    cli_main()
