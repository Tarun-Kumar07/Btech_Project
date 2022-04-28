import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

import snntorch as snn
from snntorch import surrogate 

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_intermediate_values, plot_optimization_history
# from optuna.intergration.tensorboard import TensorBoardCallaback

from utils import Classifier, DVSGestureDataModule#, quantize

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

    lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75),"init_hidden":True}

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
                          snn.Leaky(**lif_params),
                        )

    return model

class SCNN(nn.Module):

    def __init__(self,lif_params,in_channels,out_channels,kernel,stride=1,bias=True,*args,**kwargs):
        super(SCNN,self).__init__(*args,**kwargs)

        # Need to add padding to not change the x,y dimensions
        need_padding = in_channels == out_channels and stride == 1
        padding = kernel // 2 if need_padding else 0

        conv = nn.Conv2d(in_channels,out_channels,kernel,stride,padding,bias=bias)
        leaky = snn.Leaky(**lif_params)

        self.model = nn.Sequential(conv,leaky)
    
    def forward(self,x):
        return self.model(x)

def test_scnn():
    lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75),"init_hidden":True}
    scnn = SCNN(2,4,5,**lif_params)
    input_ = torch.rand((1,2,10,10))
    out_ = scnn(input_)
    print(out_.shape)

class SLINEAR(nn.Module):

    def __init__(self,lif_params,in_features,out_features,bias=True,*args,**kwargs):
        super(SLINEAR,self).__init__(*args,**kwargs)
        linear = nn.Linear(in_features,out_features,bias)
        leaky = snn.Leaky(**lif_params)
        self.model = nn.Sequential(linear,leaky)

    def forward(self,x):
        # x = self.linear(x)
        # x = self.leaky(x)

        return self.model(x)


class CustomSNN(nn.Module):
    def __init__(self,conv_config:list,linear_config:list,in_channels:int,dropout_rate:float,depth:float,width:float,lif_params:dict):
        super(CustomSNN,self).__init__()
        self._in_channels = in_channels
        self.depth_factor = depth 
        self.width_factor = width

        self.lif_params  = lif_params 
        self.dropout_rate = dropout_rate

        conv_layers = self._create_conv(conv_config)
        linear_layers  = self._create_linear(linear_config)

        model = nn.Sequential(*conv_layers,nn.Flatten(),*linear_layers)
        # self._modules = deepcopy(model._modules)
        self.model = model


        self.log_params = {"conv_config":conv_config,"linear_config":linear_config,"in_channels":in_channels,"depth":depth,width:width,"lif_params":lif_params}

    def _create_conv_blocks(self,conv_params) -> nn.Module:
        kernel,out_channels,stride,repeat = conv_params

        repeat = int(repeat * self.depth_factor)
        out_channels = int(out_channels * self.width_factor)

        conv = SCNN(self.lif_params,self._in_channels,out_channels,kernel,stride)
        layers = [conv]
        
        self._in_channels = out_channels

        for _ in range(repeat - 1):
            layers.append(SCNN(self.lif_params,out_channels,out_channels,kernel))

        layers.append(nn.Dropout(self.dropout_rate))
       
        return nn.Sequential(* layers)
    
    def _create_conv(self,conv_config:list)->nn.Module:
        layers = []
        for conv_params in conv_config:
            layers.append(self._create_conv_blocks(conv_params))
        return nn.Sequential(* layers)

    def _create_linear(self,linear_config:list) -> nn.Module:
        layers = []
        for in_features,out_features in linear_config:
            layers.append(SLINEAR(self.lif_params,in_features,out_features))

        return nn.Sequential(* layers)
    

    def forward(self,x):
       return self.model(x) 


def cal_linear_features(in_features,conv_config):
    out_ = 0
    in_ = in_features
    channels = 0
    for kernel,out_channel,stride,repeat in conv_config:
        out_ = int((in_-kernel)/stride) + 1
        in_ = out_
        channels = out_channel
       
    return out_* out_* channels

# def get_model(num_classes):
#     conv1 = nn.Conv2d(2,4,5,2,2)
#     conv2 = nn.Conv2d(4,8,5,2,2)
#     conv3 = nn.Conv2d(8,8,3,2,1)
#     conv4 = nn.Conv2d(8,16,3,2,1)
#     dropout = nn.Dropout2d()   
#     linear = nn.Linear(1024,num_classes)   

#     lif_params = {"beta":0.7,"threshold":0.3,"spike_grad":surrogate.fast_sigmoid(75),"init_hidden":True}

#     model = nn.Sequential(
#                           conv1, 
#                           snn.Leaky(**lif_params),
#                           conv2,
#                           snn.Leaky(**lif_params),
#                           conv3,
#                           snn.Leaky(**lif_params),
#                           conv4,
#                           snn.Leaky(**lif_params),
#                           dropout,
#                           nn.Flatten(),
#                           linear,
#                           snn.Leaky(**lif_params),
#                         )

#     return model

def get_objective(conv_config,linear_config,gpus) :

    def objective(trial:optuna.trial.Trial) -> float:

        slope = trial.suggest_int("slope",20,100)
        lif_params = {
                "beta" : trial.suggest_float("beta",0.5,1),
                "threshold" : trial.suggest_float("threshold",0.3,3),
                "spike_grad" : surrogate.fast_sigmoid(slope) ,
                "init_hidden" : True
                }

        learning_rate = trial.suggest_float("learning_rate",1e-4,1e-2,log=True)
        dropout_rate = trial.suggest_float("dropout_rate",0,0.5)

        model = CustomSNN(conv_config,linear_config,2,dropout_rate,1.0,1.0,lif_params)
        logger = TensorBoardLogger("./logs",name="custom_model_hp",log_graph=True)

        clf = Classifier(model,learning_rate=learning_rate)
        dm = DVSGestureDataModule("./data") 
        trainer = pl.Trainer(logger = logger,
                             max_epochs=50,
                             gpus = gpus,
                             fast_dev_run=False,
                             callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
                             strategy = DDPPlugin(find_unused_parameters=False)
                            )
        trainer.fit(clf,dm)
        # trainer.test(clf,dm)

        hparams = {"conv_config":conv_config,"linear_config":linear_config,"lif_params":lif_params}
        logger.log_hyperparams(hparams)

        return trainer.callback_metrics["val_acc"].item()

    return objective

# def test_dm():
#     dm = DVSGestureDataModule("./data") 
#     dm.setup()

#     train_dl = dm.train_dataloader()

#     for x,y in train_dl:
#         print(x[0][0])
#         break
    
# def objective_test(trial):
#     x = trial.suggest_float('x', -10, 10)
#     y = trial.suggest_float('y', -10, 10)

#     return (x - 2) ** 2 - y**2 

def cli_main():
    conv_config = [ #kernel ,out_channels, stride, repeat 
                    [7,8,2,2],
                    [5,16,2,2],
                    [3,32,2,2],
                    [3,64,2,2],
                  ]

    linear_features = cal_linear_features(128,conv_config)
    # print(linear_features)

    linear_config = [ #in_features, out_features
                      [linear_features,11]
            ]


    gpus = 3 # torch.cuda.device_count()

    objective = get_objective(conv_config,linear_config,gpus)

    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction="maximize",pruner=pruner)
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    with open("./logs/custom_model_hp/hparm.txt","a+") as f:
        f.write(str(best_params))

    logger = TensorBoardLogger("./logs",name="custom_model",log_graph=True)
    optuna_visulaizers = [plot_parallel_coordinate, plot_param_importances, plot_intermediate_values, plot_optimization_history]

    for v in optuna_visulaizers:
        filename = "./logs/custom_model_hp/" +  v.__name__+".html"
        v(study).write_html(full_html=True,file=filename)

    
    # best_params = {
    #         "beta" : 0.7,
    #         "threshold" : 0.3,
    #         "spike_grad" : surrogate.fast_sigmoid(),
    #         "learning_rate" : 1e-3,
    #         "dropout_rate" : 0.8

    #         }

    ##After tunining
    lif_params = {
                  "beta":best_params["beta"],
                  "threshold":best_params["threshold"],
                  "spike_grad":surrogate.fast_sigmoid(best_params["slope"]),
                  "init_hidden":True
                  }


    model = CustomSNN(conv_config,linear_config,2,best_params["dropout_rate"],1.0,1.0,lif_params)
    dm = DVSGestureDataModule("./data")
    clf = Classifier(model,best_params["learning_rate"])
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(logger=logger,
                         gpus=gpus,
                         max_epochs=200,
                         callbacks=[lr_monitor],
                         strategy = DDPPlugin(find_unused_parameters=False),
                         )

    trainer.fit(clf,dm)
    trainer.test(clf,dm)

if __name__ == "__main__":
    cli_main()
    # test()

