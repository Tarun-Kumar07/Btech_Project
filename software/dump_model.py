from models import get_model
from utils import Classifier,DVSGestureDataModule
import os
from fxpmath import Fxp

FXP_REF = Fxp(None,signed=True,n_int=2,n_frac=7)

def write_conv_layer(filters,file):
    n_kernel,n_depth,n_x,n_y = filters.shape
    # file.write(f"reg [10:0] filter [{n_kernel*n_depth-1}:0][{n_x-1}:0][{n_y-1}:0];\n")
    for k in range(n_kernel):
        for d in range(n_depth):  
            for x in range(n_x):
                for y in range(n_y):
                    fxp = Fxp(filters[k][d][x][y].numpy(),like=FXP_REF)
                    binary = fxp.bin()
                    file.write(f"filter5[{n_depth*k+d}][{x}][{y}] = 10'b{binary};\n")

def write_linear_layer(weights,file):
    n_rows,n_cols = weights.shape
    # file.write(f"reg [10:0] filter [{n_rows-1}:0][{n_cols-1}:0];\n")
    for r in range(n_rows):
        for c in range(n_cols):  
            fxp = Fxp(weights[r][c].numpy(),like=FXP_REF)
            binary = fxp.bin()
            file.write(f"filter[{r}][{c}] = 10'b{binary};\n")

def write_bias(bias,file):
    n_bias = bias.shape[0]

    # file.write(f"reg [10:0] bias [{n_bias - 1}:0];\n")
    for b in range(n_bias):
        fxp = Fxp(bias[b].numpy(),like=FXP_REF)
        binary = fxp.bin()
        file.write(f"bias5[{b}] = 10'b{binary};\n")

def dump_data(path="./data",out="./data_out"):
    dm = DVSGestureDataModule(path)
    dm.setup()
    train_dl = dm.train_dataloader()

    data,labels = None,None
    for x,y in train_dl:
        data,labels = x,y
        break

    os.makedirs(out,exist_ok=True)

    count = 0
    for i in range(1):
        filepath = os.path.join(out, "data_" + str(count))

        with open(filepath,"a") as file:

            for t in range(1):
                for c in range(data.shape[2]):
                    for x in range(data.shape[3]):
                        for y in range(data.shape[4]):
                            fxp = Fxp(data[i][t][c][x][y].numpy(),like=FXP_REF)
                            binary = fxp.bin()
                            file.write(f"{binary}\n")
                file.write("\n")


        count += 1

    


def dump_model(model=get_model(11),path="./logs/old_model/checkpoints/epoch=399-step=6399.ckpt",out="./out"):

    os.makedirs(out,exist_ok=True)
    model = Classifier.load_from_checkpoint(checkpoint_path=path,backbone=model)
    state_dict  = model.state_dict()

    weight_count = 0
    bias_count = 0
    for key in state_dict.keys():
        layer_name = f"layer_{weight_count}.txt"
        fp = os.path.join(out,layer_name)
        
        if "weight" in key:
            with open(fp,"a+") as f: 
                print(f"{weight_count} : {state_dict[key].shape}")
                if len(state_dict[key].shape)==4:
                    write_conv_layer(state_dict[key],f)
                else:
                    write_linear_layer(state_dict[key],f)
                weight_count+=1
        
        layer_name = f"layer_{bias_count}.txt"
        fp = os.path.join(out,layer_name)
        if "bias" in key:
            with open(fp,"a+") as f: 
                print(f"{bias_count} : {(state_dict[key].shape[0])}")
                write_bias(state_dict[key],f)
                bias_count+=1


def test():
    fxp_ref = Fxp(None,signed=True,n_int=2,n_frac=7)
    beta = Fxp(0.7,like=fxp_ref).bin()
    thresh = Fxp(0.3,like=fxp_ref).bin()
    print(f"{beta = } {thresh = }")


if __name__ == "__main__":
    # test()
    # dump_data()
    dump_model()
