import pytorch_lightning as pl
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate, utils
# import snntorch.functional as SF

from .base import BaseClassifier

class SCNN(BaseClassifier):
    def __init__(self,num_classes) -> None:
        self.conv1 = nn.Conv2d(2,4,5,2,2)
        self.conv2 = nn.Conv2d(4,8,5,2,2)
        self.conv3 = nn.Conv2d(8,8,3,2,1)
        self.conv4 = nn.Conv2d(8,16,3,2,1)
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(1024,num_classes)   
        self.backpass = surrogate.fast_sigmoid(75)
        self.lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":self.backpass,"init_hidden":True}
        self.num_classes = num_classes

        self.model = nn.Sequential(
                              self.conv1, 
                              snn.Leaky(**self.lif_params),
                              self.conv2,
                              snn.Leaky(**self.lif_params),
                              self.conv3,
                              snn.Leaky(**self.lif_params),
                              self.conv4,
                              snn.Leaky(**self.lif_params),
                              self.dropout,
                              nn.Flatten(),
                              self.linear,
                              snn.Leaky(**self.lif_params,output=True),
                            )

        super().__init__()

    def forward(self,x):
        # x.unsqueeze_(2)
        time_steps = x.shape[0]
        x = x.swapaxes(0,1) 
        x = x.float() #weights of convolution layers are in  floats

        utils.reset(self.model)
        spk_rec = []
        for i in range(time_steps):
            spk_out,mem_out = self.model(x[i])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=32)

        return {"optimizer":optimizer,"lr_scheduler":scheduler}
        # return optimizer
