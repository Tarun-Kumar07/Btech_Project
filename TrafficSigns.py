import os
import shutil
import torch
import torch.nn as nn

import pytorch_lightning as pl

from torchmetrics.functional import accuracy
from torchmetrics import ConfusionMatrix 
from norse.torch.module import Lift
from norse.torch import SequentialState, PoissonEncoder, LIF, LIFParameters

from data_modules import TrafficSignsDataModule

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

        self.conv1 = nn.Conv2d(1,16,5,2)
        self.conv2 = nn.Conv2d(16,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.maxPool = nn.MaxPool2d(2)
        self.linear = nn.Linear(128*3*3,self.num_classes)

        self.model = SequentialState(
                                  PoissonEncoder(self.seq_length,self.fmax), 
                                  Lift(self.conv1), #48
                                  Lift(self.maxPool), #24
                                  LIF(self.lif_params),
                                  Lift(self.conv2), #21
                                  Lift(self.maxPool), #11
                                  LIF(self.lif_params),
                                  Lift(self.conv3), #6
                                  Lift(self.maxPool), #3
                                  LIF(self.lif_params),
                                  Lift(nn.Flatten()), #32*3*3 
                                  Lift(self.linear),
                                  LIF(self.lif_params),
                                  )

    def forward(self,x):
        out,state = self.model(x) 

        #Compute spiking rate by summing across the time dimension (first dimension)
        spikes = torch.sum(out,0)

        #Convert to logits and apply cross entropy loss
        logits = nn.functional.softmax(spikes,dim=1)

        return logits


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

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
        loss = nn.functional.nll_loss(logits,target = y)
        acc = accuracy(logits,y)
        return {"loss":loss,"acc":acc} 

    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = nn.functional.nll_loss(logits,target = y)

        self.val_confusion_matix(logits,y)
        acc = accuracy(logits,y)
        return {"loss":loss,"acc":acc} 

      
    def test_step(self,batch,batch_idx):
        x,y = batch
        logits = self(x)

        loss = nn.functional.nll_loss(logits,target = y)

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
        df_cm = pd.DataFrame(cm.numpy(), index = range(self.num_classes), columns=range(self.num_classes))
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
    dm = TrafficSignsDataModule("../TrafficSigns",rearrange=False,num_worker=cpus)

    params = LIFParameters()
    snn = SNN(seq_length=32,num_classes = 58,lif_params=params,fmax=1000)

    trainer = pl.Trainer(gpus=gpus,max_epochs=10,fast_dev_run=False)

    # rand_input = torch.rand((32,1,100,100))
    # y,_ = snn(rand_input)
    # lr_finder = trainer.tuner.lr_find(snn,dm,min_lr=1e-2)
    # print(lr_finder.results)

    # trainer.fit(snn,dm)

    trainer.test(snn,dm)

    

if __name__ == "__main__":
    main()
