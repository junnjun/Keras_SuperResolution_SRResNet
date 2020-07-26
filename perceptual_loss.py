from keras.applications.vgg19 import VGG19
from keras.layers import Input


# get VGG network
def get_VGG19(input_size):
    vgg_inp = Input(input_size)
    vgg = VGG19(include_top=False, input_tensor=vgg_inp)
    for l in vgg.layers: l.trainalbe = False

    vgg_outp = vgg.get_layer('block2_conv2').output  # which layer you want to choose

    return vgg_inp, vgg_outp


