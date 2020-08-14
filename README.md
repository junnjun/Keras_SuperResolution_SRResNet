# 2D super-resolution on cardiac MR images

A Keras implementation of super-resolution ResNet from ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802), as a part of a master thesis project "Super-resolving cardiac MR images using deep learning" at Linköping University.

A PyTorch implementation can be found [here](https://github.com/junnjun/PyTorch_SuperResolution_SRResNet) 

## Dataset

2D balanced-ssfp slices were used for training/inference. 2D slices were obtained from cine balanced-ssfp volume with spatial resolution of 1 x 1 x 8 mm^3. Obtained 2D slices are used as a hight-resolution target for training the network. Low-resolution input was created by downsampling with bicubic interpolation and adding Gaussian blurring with sigma = 1.0. 

## Network architecture and more
![SRResNet](./images/SRResNet.png)
Super-resolution ResNet (SRResNet) is a generator network of super-resolution generative adversarial network (SRGAN). SRResNet is composed of 16 residual blocks with local skip connections and one global skip connection, and 2 upscaling blocks with MSE as loss function. The network adopts pixel shuffler as its upscaling method, where convolution is performed before upscaling in each block.

![Pixelshuffler](./images/pixelshuffler.png)
[Pixel-shuffler](https://arxiv.org/abs/1609.05158) upscales the input from the previous layer by rearranging the pixel from its feature map. With the input size of H(height) x W(width) x C(channel size), it first increases the number of its channels by r^2 where r denotes the upscaling factor. Then it performsperiodic shuffling which maps the input space from depth <img> tag: <img src="https://cdn.mathpix.com/snip/images/tq_x-XP05elVow9VIOdlsAhV-LRDpQo6KAo9GZDjgVA.original.fullsize.png" /> to space <img> tag: <img src="https://cdn.mathpix.com/snip/images/SE49efYAu5BQ3ZNiZUZ4jcLP6i26im-qT1dWEz2xGZk.original.fullsize.png" />.

Hence, in this project following experiments were of interest :  first,  loss function <code> <b>perceptual loss VS MSE loss</b> </code> and second, upscaling method (upscaling factor x4) <code> <b> pixel-shuffler VS nearest-neighbor interpolation</b> </code>. Also, a different order of sequence of layers in the upscaling block is tested (Model 1 vs Model 5). The proposed combination of the network architecture of SRResNet from the paper is listed under Model 1.

|         |  Loss function  | Upscaling method |    Sequence of Layer    |
|:-------:|:---------------:|:----------------:|:-----------------------:|
| Model 1 |     MSE loss    |  Pixel-shuffler  | Convolution then Resize |
| Model 2 | Perceptual loss |  Pixel-shuffler  | Convolution then Resize |
| Model 3 |     MSE loss    | NN interpolation | Convolution then Resize |
| Model 4 | perceptual loss | NN interpolation | Convolution then Resize |
| Model 5 |    Mse loss     |  Pixel-shuffler  | Resize then Convolution |

## Training detail
Training was performed on a workstation with a 3.6GHz, 6-core processor with 64GB RAM, NVIDIA Quadro P6000 GPU.

## Usage
