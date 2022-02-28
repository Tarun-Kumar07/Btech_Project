import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from torchmetrics import ConfusionMatrix

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate    
from snntorch import utils

from data_modules import DVSGestureDataModule

class Network(pl.LightningModule):
    def __init__(self,num_classes,time_steps) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv2d(2,4,5,2,2)
        self.conv2 = nn.Conv2d(4,8,5,2,2)
        self.conv3 = nn.Conv2d(8,8,3,2,1)
        self.conv4 = nn.Conv2d(8,16,3,2,1)
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(1024,num_classes)   
        self.backpass = surrogate.fast_sigmoid(75)
        self.lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":self.backpass,"init_hidden":True}

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


        #Meterics
        self.accuracy = SF.accuracy_rate
        self.loss = SF.ce_rate_loss()
        self.val_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')
        self.test_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')

    def forward(self,x):
        # x.unsqueeze_(2)
        x = x.swapaxes(0,1) 
        x = x.float() #weights of convolution layers are in  floats

        utils.reset(self.model)
        spk_rec = []
        for i in range(self.time_steps):
            spk_out,mem_out = self.model(x[i])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=32)

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


# class Dataset(pl.LightningDataModule):
#     def __init__(self,data_dir,num_steps,batch_size,num_workers=4,val_ratio=0.1) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.num_steps = num_steps 
#         self.val_ratio = val_ratio
#         self.num_workers = num_workers
#         self.batch_size = batch_size

#     def setup(self, stage) -> None:

#         if stage == "fit" or stage is None:
#             dataset = spikedata.DVSGesture(root=self.data_dir,train=True,num_steps=self.num_steps)
            
#             total = len(dataset)
#             val_size = int(total * self.val_ratio)
#             train_size = total - val_size

#             self.train_ds , self.val_ds = random_split(dataset,[train_size,val_size])

#         if stage == "test" or stage is None:
#             self.test_ds = spikedata.DVSGesture(root=self.data_dir,train=False,num_steps=self.num_steps)

#     def train_dataloader(self) :
#         return DataLoader(self.train_ds,num_workers=self.num_workers,batch_size=self.batch_size,pin_memory=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_ds,num_workers=self.num_workers,batch_size=self.batch_size,pin_memory=True)

#     def test_dataloader(self) :
#         return DataLoader(self.test_ds,num_workers=self.num_workers,batch_size=self.batch_size,pin_memory=True)

def main():
    root_path = "../dvs128"
    num_classes = 11
    time_steps = 150
    num_workers = 4
    gpus  = torch.cuda.device_count()
    batch_size = 16

    snn = Network(num_classes=num_classes,time_steps=time_steps)
    dm = DVSGestureDataModule(data_dir=root_path,num_workers=num_workers,batch_size=batch_size,n_time_bins=time_steps)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch') 
    
    trainer = pl.Trainer(callbacks=[lr_monitor],gpus=gpus,max_epochs=256,fast_dev_run=False,gradient_clip_val=5,strategy=DDPPlugin(find_unused_parameters=False))


    trainer.fit(snn,dm)

    trainer.test(snn,dm)

if __name__ == "__main__":
    main()
