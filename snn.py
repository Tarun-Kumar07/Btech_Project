import os
import shutil
import torch
import torch.nn as nn

import pytorch_lightning as pl
import numpy as np

from torchmetrics.functional import accuracy
from torchmetrics import ConfusionMatrix 
from norse.torch.module import Lift, LConv2d
from norse.torch import SequentialState, PoissonEncoder, LIF, LIFParameters

from data_modules import DVSGestureDataModule

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.loss = nn.CrossEntropyLoss()
        self.conv1 = LConv2d(1,4,5,2,2)
        self.conv2 = LConv2d(4,8,5,2,2)
        self.conv3 = LConv2d(8,8,3,2,1)
        self.conv4 = LConv2d(8,16,3,2,1)
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(1024,self.num_classes)
        
        self.model = SequentialState(
                                  # PoissonEncoder(self.seq_length,self.fmax), 
                                  self.conv1, #48
                                  LIF(self.lif_params),
                                  self.conv2, #21
                                  LIF(self.lif_params),
                                  self.conv3, #6
                                  LIF(self.lif_params),
                                  self.conv4, #6
                                  LIF(self.lif_params),
                                  self.dropout,
                                  Lift(nn.Flatten()), #32*3*3 
                                  Lift(self.linear),
                                  LIF(self.lif_params),
                                  )

    def forward(self,x):
        
        # x = x[:,:,None,:,:]
        # x = np.swapaxes(x,1,0) # Making time axis as outer axis

        x.unsqueeze_(2)
        x = x.swapaxes(0,1) 
        x = x.float() #weights of convolution layers are in  floats

        # print(x.shape)
        out = self.model(x) 
        print(out.shape)

        # print("Applying conv1")
        # x = self.conv1(x)
        # print(x.shape)
        # print("Applying conv2")
        # x = self.conv2(x)
        # print(x.shape)
        # print("Applying conv3")
        # x = self.conv3(x)
        # print(x.shape)
        # print("Applying conv4")
        # x = self.conv4(x)
        # print(x.shape)
        # print("Applying Dropout")
        # x = self.dropout(x)
        # print(x.shape)
        # print("After flatten")
        # x = Lift(nn.Flatten())(x)
        # print(x.shape)
        # print("After Linear")
        # x = Lift(self.linear)(x)
        # print(x.shape)

        #Compute spiking rate by summing across the time dimension (first dimension)
        spikes = torch.sum(out,0)

        #Convert to logits and apply cross entropy loss
        logits = nn.functional.softmax(spikes,dim=1)

        return logits


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.7)

        return {"optimizer":optimizer,"lr_scheduler":scheduler}

    #def _step(self,batch):
    #    # Return the loss
    #    x,y = batch
    #    # x shape => (batch_size,1,100,100)

    #    #Compute Negative log likelihood loss and accuracy
    #    logits = self(x)
    #    loss = nn.functional.nll_loss(logits,target = y)

    #    return loss 

    def training_step(self,batch,batch_idx):
        # Return the loss
        x,y = batch
        # x shape => (batch_size,1,100,100)

        #Compute Negative log likelihood loss and accuracy
        logits = self(x)
        loss = self.loss(logits,y)
        acc = accuracy(logits,y)
        return {"loss":loss,"acc":acc} 

    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = self.loss(logits,y)

        self.val_confusion_matix(logits,y)
        acc = accuracy(logits,y)
        return {"loss":loss,"acc":acc} 

      
    def test_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = self.loss(logits,y)

        self.test_confusion_matix(logits,y)
        acc = accuracy(logits,y)
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
    cpus = os.cpu_count()
    dm = DVSGestureDataModule(num_workers=cpus,batch_size=16)

    params = LIFParameters(alpha=3,v_th=0.3,v_leak=0.7,method="heavy")
    snn = SNN(seq_length=32,num_classes = 11,lif_params=params,fmax=1000)


    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(callbacks=[lr_monitor],gpus=gpus,max_epochs=20,fast_dev_run=True,gradient_clip_val=5)

    # rand_input = torch.rand((16,1,128,128))
    # y,_ = snn(rand_input)
    # lr_finder = trainer.tuner.lr_find(snn,dm,min_lr=1e-5)
    # print(lr_finder.results)

    trainer.fit(snn,dm)

    trainer.test(snn,dm)

    

if __name__ == "__main__":
    main()
