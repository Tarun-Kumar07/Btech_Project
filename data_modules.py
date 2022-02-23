
import pytorch_lightning as pl

import tonic
from tonic.datasets import DVSGesture
from tonic.transforms import Compose,ToFrame, ToVoxelGrid

from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder 

class DVSGestureDataModule(pl.LightningDataModule):

    def __init__(self,data_dir="../dvs128",batch_size=16,num_workers=4,val_ratio=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.val_ratio = val_ratio

        sensor_size = DVSGesture.sensor_size
        self.transforms = Compose([
            ToVoxelGrid(sensor_size=sensor_size,n_time_bins=150)
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
        return DataLoader(self.train_ds,batch_size = self.batch_size, num_workers= self.num_worker)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size = self.batch_size,num_workers= self.num_worker)

    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size = self.batch_size,num_workers= self.num_worker)


# class TrafficSignsDataModule(pl.LightningDataModule):

#     def __init__(self,data_dir,rearrange=False,num_worker=1,intensity=10,val_ratio=0.1,batch_size=32):

#         super(TrafficSignsDataModule,self).__init__()
#         # self.dims = (1,100,100)
#         self.val_ratio = val_ratio
#         self.batch_size = batch_size
#         self.data_dir = data_dir
#         self.rearrange = rearrange
#         self.num_worker = num_worker
#         self.transform = transforms.Compose([
#           transforms.Resize((100,100)),
#           transforms.Grayscale(),
#           transforms.ToTensor(),
#           transforms.Lambda(lambda x: x * intensity),                                    
#         ])
      
  
#     def prepare_data(self):
#         # Download the dataset at self.data_dir
#         # Rearrange function
#         def rearrange_dataset(src_folder,dest_folder,annotation_file):
#           with open(annotation_file) as f:
#             for line in f:
#               splitted = line.split(";")
#               image_name = splitted[0]
#               label = splitted[-2]

#               label_folder = os.path.join(dest_folder,label)
#               os.makedirs(label_folder,exist_ok=True)

#               src_path = os.path.join(src_folder,image_name)

#               shutil.copy2(src_path,label_folder)

        
#         train_src = os.path.join(self.data_dir,"tsrd-train")
#         train_dest = os.path.join(self.data_dir,"Train")
#         train_annotation = os.path.join("TSRD-Train Annotation/TsignRecgTrain4170Annotation.txt")
#         test_src = os.path.join("TSRD-Test")    
#         test_dest = os.path.join(self.data_dir,"Test")
#         test_annotation = os.path.join("TSRD-Test Annotation/TsignRecgTest1994Annotation.txt")

#         if self.rearrange:
#           rearrange_dataset(train_src,train_dest,train_annotation)
#           rearrange_dataset(test_src,test_dest,test_annotation)

#         self.train_dir = train_dest
#         self.test_dir = test_dest
#         self.num_classes = len(os.listdir(train_dest))

#     def setup(self,stage=None):

#         if stage == "fit" or stage is None:
#           dataset = ImageFolder(self.train_dir,transform = self.transform)

#           size = len(dataset)
#           train_size = int((1-self.val_ratio)*size)
#           val_size = size - train_size

#           self.train_ds , self.val_ds = random_split(dataset,[train_size,val_size])
        
#         if stage == "test" or stage is None:
#           self.test_ds = ImageFolder(self.test_dir,transform = self.transform)


#     def train_dataloader(self):
#         return DataLoader(self.train_ds,batch_size = self.batch_size, num_workers= self.num_worker)

#     def val_dataloader(self):
#         return DataLoader(self.val_ds,batch_size = self.batch_size,num_workers= self.num_worker)

#     def test_dataloader(self):
#         return DataLoader(self.test_ds,batch_size = self.batch_size,num_workers= self.num_worker)

def test_dvs128():
    path = "../dvs128"
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
