import torch

import pytorch_lightning as pl

from snntorch import utils
import snntorch.functional as SF

from tonic.datasets import DVSGesture
from tonic.transforms import Compose,ToFrame, ToVoxelGrid
from torch.utils.data import random_split, DataLoader


class Classifier(pl.LightningModule):

    def __init__(self,backbone:torch.nn.Module,learning_rate=1e-3) -> None:
        super(Classifier,self).__init__()
        self.save_hyperparameters()
        self.backbone = backbone 

        #Metrics
        self.accuracy = SF.accuracy_rate
        self.loss = SF.ce_rate_loss()
        # self.val_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')
        # self.test_confusion_matix = ConfusionMatrix(num_classes=num_classes,normalize='true')

    def forward(self,x):
        # x.unsqueeze_(2)
        x = x.swapaxes(0,1) 
        x = x.float() #weights of convolution layers are in  floats

        utils.reset(self.backbone)
        spk_rec = []
        time_steps = x.shape[0]

        for i in range(time_steps):
            spk_out,mem_out = self.backbone(x[i])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.hparams.learning_rate)

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
        if self.current_epoch == 0:
            sample_input = torch.rand(-1,2,128,128)
            self.logger.experiment.add_graph(self.backbone,sample_input)

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


class DVSGestureDataModule(pl.LightningDataModule):

    def __init__(self,data_dir,batch_size=16,n_time_bins=150,num_workers=4,val_ratio=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.val_ratio = val_ratio
        self.persistent = True
        sensor_size = DVSGesture.sensor_size
        self.transforms = Compose([
            ToFrame(sensor_size=sensor_size,n_time_bins=n_time_bins)
            ])

    def setup(self,stage=None):
        if stage == "fit" or stage is None:
          dataset = DVSGesture(save_to = self.data_dir, train=True,transform=self.transforms) 

          size = len(dataset)
          train_size = int((1-self.val_ratio)*size)
          val_size = size - train_size

          self.train_ds , self.val_ds = random_split(dataset,[train_size,val_size])
        
        if stage == "test" or stage is None:
          self.test_ds = DVSGesture(save_to = self.data_dir, train=False,transform=self.transforms) 

    def train_dataloader(self):
        return DataLoader(self.train_ds,batch_size = self.batch_size, num_workers= self.num_worker, persistent_workers=self.persistent)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size = self.batch_size,num_workers= self.num_worker, persistent_workers=self.persistent)

    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size = self.batch_size,num_workers= self.num_worker, persistent_workers=self.persistent)

