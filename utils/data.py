import pytorch_lightning as pl

from tonic.datasets import DVSGesture
from tonic.transforms import Compose,ToFrame, ToVoxelGrid

from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder 

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


def test_dvs128():
    path = "../../data/"
    dm = DVSGestureDataModule(data_dir=path,batch_size=32,num_workers=8,val_ratio=0.1)
    dm.setup()
    train = dm.train_dataloader()
    for x,y in train:
        print(x.shape)
        # print(x[0])
        print(y)
        break


    print(DVSGesture.sensor_size)

if __name__ == "__main__":
    test_dvs128()
