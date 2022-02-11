import os
import shutil
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.functional import accuracy
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder 
from norse.torch.module import Lift
from norse.torch import SequentialState, PoissonEncoder, LIF, LIFParameters


class SNN(pl.LightningModule):

    def __init__(self,seq_length,num_classes,lif_params,fmax,learning_rate=1e-3):
        super().__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.lif_params = lif_params
        self.fmax = fmax 
        self.learning_rate = learning_rate

        self.model = SequentialState(
                                      PoissonEncoder(self.seq_length,self.fmax), 
                                      Lift(nn.Conv2d(1,8,5,2)), #48
                                      Lift(nn.MaxPool2d(2)), #24
                                      LIF(self.lif_params),
                                      Lift(nn.Conv2d(8,16,5)), #21
                                      Lift(nn.MaxPool2d(2)), #11
                                      LIF(self.lif_params),
                                      Lift(nn.Conv2d(16,32,5)), #6
                                      Lift(nn.MaxPool2d(2)), #3
                                      LIF(self.lif_params),
                                      Lift(nn.Flatten()), #32*3*3 
                                      Lift(nn.Linear(32*3*3,self.num_classes)),
                                      LIF(self.lif_params),
                                      )
   

    def forward(self,x):
        return self.model(x) 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _step(self,batch):
        # Return the loss
        x,y = batch
        # x shape => (batch_size,1,100,100)
        out,state = self(x)

        #Compute spiking rate by summing across the time dimension (first dimension)
        spikes = torch.sum(out,0)

        #Convert to logits and apply cross entropy loss
        logits = nn.functional.softmax(spikes,dim=1)

        #Compute Negative log likelihood loss and accuracy
        loss = nn.functional.nll_loss(logits,target = y)
        acc = accuracy(logits,y)

        pbar = {"acc":acc}
        return {"loss":loss,"acc":acc,"progress_bar":pbar}

    def training_step(self,batch,batch_idx):
        return self._step(batch) 

    def validation_step(self,batch,batch_idx):
        return self._step(batch,) 
      
    def test_step(self,batch,batch_idx):
        return self._step(batch)


    def _epoch_end(self,step_outputs):
        avg_loss = torch.Tensor([x["loss"] for x in step_outputs]).mean()
        avg_acc = torch.Tensor([x["acc"] for x in step_outputs]).mean()

        return avg_loss, avg_acc

    def training_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        print(loss , acc)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

    def validation_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_epoch_end(self,step_outputs):
        loss, acc = self._epoch_end(step_outputs)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

class TrafficSignsDataModule(pl.LightningDataModule):

  def __init__(self,data_dir,rearrange=False,num_worker=1,intensity=10,val_ratio=0.1,batch_size=32):

    super(TrafficSignsDataModule,self).__init__()
    # self.dims = (1,100,100)
    self.val_ratio = val_ratio
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.rearrange = rearrange
    self.num_worker = num_worker
    self.transform = transforms.Compose([
      transforms.Resize((100,100)),
      transforms.Grayscale(),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x * intensity),                                    
    ])
  
  
  def prepare_data(self):
    # Download the dataset at self.data_dir
    # Rearrange function
    def rearrange_dataset(src_folder,dest_folder,annotation_file):
      with open(annotation_file) as f:
        for line in f:
          splitted = line.split(";")
          image_name = splitted[0]
          label = splitted[-2]

          label_folder = os.path.join(dest_folder,label)
          os.makedirs(label_folder,exist_ok=True)

          src_path = os.path.join(src_folder,image_name)

          shutil.copy2(src_path,label_folder)

    
    train_src = os.path.join(self.data_dir,"tsrd-train")
    train_dest = os.path.join(self.data_dir,"Train")
    train_annotation = os.path.join("TSRD-Train Annotation/TsignRecgTrain4170Annotation.txt")
    test_src = os.path.join("TSRD-Test")    
    test_dest = os.path.join(self.data_dir,"Test")
    test_annotation = os.path.join("TSRD-Test Annotation/TsignRecgTest1994Annotation.txt")

    if self.rearrange:
      rearrange_dataset(train_src,train_dest,train_annotation)
      rearrange_dataset(test_src,test_dest,test_annotation)

    self.train_dir = train_dest
    self.test_dir = test_dest
    self.num_classes = len(os.listdir(train_dest))

  def setup(self,stage=None):

    if stage == "fit" or stage is None:
      dataset = ImageFolder(self.train_dir,transform = self.transform)

      size = len(dataset)
      train_size = int((1-self.val_ratio)*size)
      val_size = size - train_size

      self.train_ds , self.val_ds = random_split(dataset,[train_size,val_size])
    
    if stage == "test" or stage is None:
      self.test_ds = ImageFolder(self.test_dir,transform = self.transform)


  def train_dataloader(self):
    return DataLoader(self.train_ds,batch_size = self.batch_size, num_workers= self.num_worker)

  def val_dataloader(self):
    return DataLoader(self.val_ds,batch_size = self.batch_size,num_workers= self.num_worker)

  def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size = self.batch_size,num_workers= self.num_worker)

def main():
    gpus = torch.cuda.device_count()
    cpus = os.cpu_count()
    dm = TrafficSignsDataModule("../TrafficSigns",rearrange=False,num_worker=cpus)

    params = LIFParameters(v_th = 0.001)
    snn = SNN(seq_length=32,num_classes = 58,lif_params=params,fmax=1000)

    trainer = pl.Trainer(gpus=gpus,max_epochs=100)

    trainer.fit(snn,dm)

    trainer.test(snn,dm)

if __name__ == "__main__":
    main()
