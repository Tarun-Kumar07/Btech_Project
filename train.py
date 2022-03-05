import pytorch_lightning as pl
import torch

from utils.models import Classifier, get_model
from utils.data import DVSGestureDataModule

dm = DVSGestureDataModule("./data/")

model = get_model(11)
optimizer = torch.optim.Adam(model.parameters())
classifier = Classifier(model,optimizer)

trainer = pl.Trainer(fast_dev_run=True)
trainer.fit(classifier,dm)
