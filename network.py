# Generator network (SRResNET)
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2


from keras.layers import Input

import keras.backend as K

from pixelshuffler import *


# Residual block
def res_block(input):
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    x = BatchNormalization(momentum=0.8)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    return add([x, input])

# (1) Efficient sub-pixel convolution(pixelshuffler) Conv -> Pixelshuffle -> PReLU
def up_block1(input, batch_size, upscale):
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    # x = Conv2D(filters=256*(upscale**2), kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    x = pixelshuffler(input_shape=(88,88,3), batch_size=batch_size, scale=upscale)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x

def up_block2(input, batch_size, upscale):
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    # x = Conv2D(filters=256*(upscale**2), kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    x = pixelshuffler(input_shape=(176,176,3), batch_size=batch_size, scale=upscale)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x


# (2) Nearest Neighbor interpolation rescaling CONV -> NN Resize -> PReLU
def up_block(input, upscale):
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    x = UpSampling2D(size=(upscale, upscale))(x)  # size:upsampling factor
    x = PReLU(shared_axes=[1, 2])(x)
    return x


def generator(batch_size, input_size=(88,88,3), upscale=4, mode='PS'):
    input = Input(input_size)
    x_init = Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), padding='same')(input)
    x = PReLU(shared_axes=[1, 2])(x_init)
    for i in range(16):
        x = res_block(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = add([x, x_init])
    if mode == 'PS':
        x = up_block1(x, batch_size=batch_size, upscale=int(upscale*0.5))
        x = up_block2(x, batch_size=batch_size, upscale=int(upscale*0.5))
    elif mode == 'NN':
        x = up_block(x, upscale=upscale*0.5)
        x = up_block(x, upscale=upscale*0.5)
    else:
        print(" please select mode PS or NN ")
    output = Conv2D(filters=3, kernel_size=(9, 9), strides=(1, 1), padding='same')(x)

    return (input, output)

