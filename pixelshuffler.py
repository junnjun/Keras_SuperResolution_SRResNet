import tensorflow as tf
from keras.layers import Lambda

"""
Subpixel convolution layer, sometimes called a pixel shuffle or phase shift.
it computes a normal convolution with stride of one and creates many filter maps which are then reorganized (from the depth of the image to the space of the image) into a larger image.

To increase the image dimensions (H, W, C) by a factor of r, we would first perform a convolution which results in an output of size (H, W, r2C) and then “shuffle the pixels” into an image of size (rH, rW, C). This way, we lose no information in the upscaling of our image.
This operation is made nearly trivial by TensorFlow’s built-in depth-to-space method.
"""

########################################################################################################################
# Implementation 1 #####################################################################################################
########################################################################################################################

def pixelshuffler(input_shape, batch_size, scale=2):

    """
    Transform input shape into sub-pixel shape
    (from H * W * C * r^2 to rH * rW * C)
    """

    def subpixel_shape(input_shape=input_shape, batch_size=batch_size):

        dim = [batch_size,
               input_shape[1] * scale,
               input_shape[2] * scale,
               int(input_shape[3]/ (scale ** 2))]

        output_shape = tuple(dim)

        return output_shape

    def pixelshuffle_upscale(x):
        # reorganized (from the depth of the image to the space of the image) into a larger image
        return tf.nn.depth_to_space(input=x, block_size=scale)

    return Lambda(function=pixelshuffle_upscale, output_shape=subpixel_shape)


########################################################################################################################
# Implementation 2 #####################################################################################################
########################################################################################################################

#def up_block_shuffler(input, upsampling_factor=2):
#    n_filters = int(input.shape[3])
#    x = Conv2D(filters=n_filters*upsampling_factor**2, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
#    x = tf.nn.depth_to_space(x, block_size=upsampling_factor)
#    x = PReLU(shared_axes=[1, 2])(x)
#    return x

