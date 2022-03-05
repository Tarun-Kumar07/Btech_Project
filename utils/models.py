import torch
import torch.nn as nn

import pytorch_lightning as pl

import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate, utils

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
                          linear,
                          snn.Leaky(**lif_params,output=True),
                        )

    return model

class Classifier(pl.LightningModule):

    def __init__(self,model:torch.nn.Module,optimizer:torch.optim.Optimizer,scheduler=None) -> None:
        super(Classifier,self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler 

        #Metrics
        self.accuracy = SF.accuracy_rate
        self.loss = SF.ce_rate_loss()
        # self.val_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')
        # self.test_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')

        self.save_hyperparameters()


    def forward(self,x):
        # x.unsqueeze_(2)
        x = x.swapaxes(0,1) 
        x = x.float() #weights of convolution layers are in  floats

        utils.reset(self.model)
        spk_rec = []
        time_steps = x.shape[0]

        for i in range(time_steps):
            spk_out,mem_out = self.model(x[i])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return {'optimizer':self.optimizer,'lr_scheduler':self.scheduler}

    def training_step(self,batch,batch_idx):
        x,y = batch
        spike_rec = self(x)
        acc = self.accuracy(spike_rec,y)
        loss = self.loss(spike_rec,y)

        return {"loss":loss,"acc":acc} 

    def validation_step(self,batch,batch_idx):
        x,y = batch
        spike_rec = self(x)
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

        # cm = self.test_confusion_matix.compute()
        # df_cm = pd.DataFrame(cm.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        # plt.figure(figsize = (10,7))
        # fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()

        # self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
        # plt.close(fig_)

    def test_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)


