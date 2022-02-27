import os
import shutil
import torch
import torch.nn as nn

import pytorch_lightning as pl
import numpy as np


from torchmetrics.functional import accuracy
from torchmetrics import ConfusionMatrix , Accuracy
from norse.torch.module import Lift, LConv2d
from norse.torch import SequentialState, PoissonEncoder, LIF, LIFParameters
import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate
from snntorch import utils

from data_modules import DVSGestureDataModule
# docker run --runtime=nvidia -it -v /home/raj/Btech_Project:/workspace/Btech_project -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 --shm-size=32G --name snn 1b65886be5f5
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


class SnnTorch():
    def __init__(self,num_classes,seq_length) -> None:
        super().__init__()
        self.seq_length = seq_length
        conv1 = nn.Conv2d(1,4,5,2,2)
        conv2 = nn.Conv2d(4,8,5,2,2)
        conv3 = nn.Conv2d(8,8,3,2,1)
        conv4 = nn.Conv2d(8,16,3,2,1)
        dropout = nn.Dropout2d()
        linear = nn.Linear(1024,num_classes)   
        lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75),"init_hidden":True}
        self.model = nn.Sequential(
                              conv1, #48
                              snn.Leaky(**lif_params),
                              conv2, #21
                              snn.Leaky(**lif_params),
                              conv3, #6
                              snn.Leaky(**lif_params),
                              conv4, #6
                              snn.Leaky(**lif_params),
                              dropout,
                              nn.Flatten(), #32*3*3 
                              linear,
                              snn.Leaky(**lif_params,output=True),
                            )
    def get_model(self):
        return self.model

    def forward(self,x):
        utils.reset(self.model)
        spk_rec = []
        for i in range(self.seq_length):
            spk_out,mem_out = self.model(x[i])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

class Norse():
    def __init__(self,num_classes) -> None:
        super().__init__()
        conv1 = LConv2d(1,4,5,2,2)
        conv2 = LConv2d(4,8,5,2,2)
        conv3 = LConv2d(8,8,3,2,1)
        conv4 = LConv2d(8,16,3,2,1)
        dropout = nn.Dropout2d()
        linear = nn.Linear(1024,num_classes)

        params = LIFParameters(alpha=3,v_th=0.3,v_leak=0.7)
        self.model = SequentialState(
                                  # PoissonEncoder(seq_length,self.fmax), 
                                  conv1, #48
                                  LIF(params),
                                  conv2, #21
                                  LIF(params),
                                  conv3, #6
                                  LIF(params),
                                  conv4, #6
                                  LIF(params),
                                  dropout,
                                  Lift(nn.Flatten()), #32*3*3 
                                  Lift(linear),
                                  LIF(params),
                                )


    def get_model(self):
        return self.model

    def forward(self,x):
        spikes , out = self.model(x)
        # print((spikes)) 
        return spikes


class SNN(pl.LightningModule):

    def __init__(self,seq_length,num_classes,lif_params,fmax,learning_rate=1e-3):
        super().__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.lif_params = lif_params
        self.fmax = fmax 
        self.learning_rate = learning_rate
        self.val_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')
        self.test_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')
        # self.val_accuracy = Accuracy(num_classes=num_classes)
        # self.test_accuracy = Accuracy(num_classes=num_classes)
        # self.loss = nn.MSELoss()
        self.accuracy = SF.accuracy_rate
        self.loss = SF.ce_rate_loss() 
        # self.lib = Norse(num_classes=num_classes)
        self.lib = SnnTorch(num_classes=num_classes,seq_length=seq_length)


    def forward(self,x):
        
        # x = x[:,:,None,:,:]
        # x = np.swapaxes(x,1,0) # Making time axis as outer axis

        x.unsqueeze_(2)
        x = x.swapaxes(0,1) 
        x = x.float() #weights of convolution layers are in  floats

        return self.lib.forward(x) 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.lib.get_model().parameters(), lr=5e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.7)

        return {"optimizer":optimizer,"lr_scheduler":scheduler}


    def training_step(self,batch,batch_idx):
        # Return the loss
        x,y = batch
        # x shape => (batch_size,1,100,100)

        #Compute Negative log likelihood loss and accuracy
        spike_rec = self(x)
        acc = self.accuracy(spike_rec,y)
        loss = self.loss(spike_rec,y)
        return {"loss":loss,"acc":acc} 

    def validation_step(self,batch,batch_idx):
        x,y = batch
        spike_rec = self(x)
        # self.val_confusion_matix(spike_rec,y)

        # y_one_hot = nn.functional.one_hot(y,num_classes=11).float()
        loss = self.loss(spike_rec,y)

        acc = self.accuracy(spike_rec,y)
        return {"loss":loss,"acc":acc} 

      
    def test_step(self,batch,batch_idx):
        x,y = batch
        spike_rec = self(x)

        loss = self.loss(spike_rec,y)

        acc = self.accuracy(spike_rec,y)
        return {"loss":loss,"acc":acc} 


    def _epoch_end(self,step_outputs):
        avg_loss = torch.Tensor([x["loss"] for x in step_outputs]).mean()
        avg_acc = torch.Tensor([x["acc"] for x in step_outputs]).mean()

        return avg_loss, avg_acc


    def training_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

    def validation_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        cm = self.test_confusion_matix.compute()
        df_cm = pd.DataFrame(cm.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()

        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
        plt.close(fig_)

    def test_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)


def main():
    gpus = torch.cuda.device_count()
    cpus = os.cpu_count() // 2
    num_time_bins = 150
    dm = DVSGestureDataModule(num_workers=cpus,batch_size=16,n_time_bins=num_time_bins)

    snn = SNN(seq_length=num_time_bins,num_classes = 11,lif_params=None,fmax=1000)


    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(callbacks=[lr_monitor],gpus=gpus,max_epochs=100,fast_dev_run=False,gradient_clip_val=5)

    # rand_input = torch.rand((16,1,128,128))
    # y,_ = snn(rand_input)
    # lr_finder = trainer.tuner.lr_find(snn,dm,min_lr=1e-5)
    # print(lr_finder.results)

    trainer.fit(snn,dm)

    trainer.test(snn,dm)

    

if __name__ == "__main__":
    main()
