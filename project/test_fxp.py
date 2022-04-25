import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import snntorch as snn

from fxpmath import Fxp

def test_conv():
    fxp_ref = Fxp(None,signed=True,n_int=2,n_frac=32)
    in_channels = 2
    out_channels = 3
    kernel_size = 3

    conv_layer = layers.Conv2D(out_channels,kernel_size,use_bias=True) 
    input_ = tf.random.normal(shape=(4,in_channels,kernel_size,kernel_size))
    out = conv_layer(input_)
    weights,bias = conv_layer.get_weights()

    fxp_weights = Fxp(weights,like=fxp_ref) 
    fxp_bias = Fxp(bias,like=fxp_ref) 

    conv_layer.set_weights([fxp_weights,fxp_bias])

    input_ = tf.random.normal(shape=(4,in_channels,kernel_size,kernel_size))
    out = conv_layer(input_)
    print(out.shape)

if __name__ == "__main__":
    test_conv()
