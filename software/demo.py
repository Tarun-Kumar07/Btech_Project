from utils import Classifier,DVSGestureDataModule
from models import CustomSNN

import snntorch
from snntorch import surrogate

import torch

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML,display


def save_html(x,path):
    x = x.swapaxes(0,1) 
    # print(x.shape)
    spike_data_sample = x[:, 0, 0]
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)

    html = anim.to_html5_video()
    with open(path,"w") as f:
        f.write(html)
    print(f"saved to {path}")
    
conv_config = [ #kernel ,out_channels, stride, repeat 
                [5,8,2,2],
                [5,16,2,2],
                [3,32,2,2],
                [3,64,2,2],
              ]

linear_features = 2304 

linear_config = [ #in_features, out_features
                  [linear_features,11]
        ]

best_params = {
        "beta" : 0.7,
        "threshold" : 0.3,
        "slope" : 92, 
        "learning_rate" : 1e-3,
        "dropout_rate" : 0.1
        }

lif_params = {
              "beta":best_params["beta"],
              "threshold":best_params["threshold"],
              "spike_grad":surrogate.fast_sigmoid(best_params["slope"]),
              "init_hidden":True
              }


clf = CustomSNN(conv_config,linear_config,2,best_params["dropout_rate"],1.0,1.0,lif_params)
model = Classifier.load_from_checkpoint("../logs_collection/logs_0505_1454/custom_model/version_0/checkpoints/epoch=399-step=6399.ckpt",backbone=clf,learning_rate=1e-3,num_classes=11)
dm = DVSGestureDataModule("./data",batch_size=1)
dm.setup()
test_ = dm.test_dataloader()
count = 0
for x,y in test_:
    path  = f"{count}.html"
    save_html(x,path)
    out_ = model(x)
    sum_ = torch.sum(out_,dim=0)
    pred = torch.argmax(sum_)
    count += 1

    print(f"True value = {y}, {pred = }")
    if count > 2:
        break
