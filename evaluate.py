import argparse
import os
from math import ceil

from network import *
from data_utility import *
from perceptual_loss import *

from keras.models import Model
from keras.layers import Lambda

parser = argparse.ArgumentParser(description='Inference 2D super-resolution ResNet')
parser.add_argument('--mode', default='PS', type=str, help='upscaling method, choose between PS or NN')
parser.add_argument('--loss', default='mse', type=str, help='loss function, choose between mse or perceptual')
parser.add_argument('--LR_input_size', default=88, type=int, help='width or height of input dim')
parser.add_argument('--test_data_dir', default='/media/saewon/Data/Saewon_thesis/Dataset/validation', type=str,
                    help='directory where LR and HR volumes are saved for training')
parser.add_argument('--load_weight_dir', default=None,
                    type=str,
                    help='which weight to load?')
parser.add_argument('--HR_folder', default='bssfp', type=str, help='folder where HR volumes are saved')
parser.add_argument('--LR_folder', default='bssfp_lowres', type=str, help='folder where LR volumes are saved')
parser.add_argument('--save_result_dir', default='/media/saewon/Data/Saewon_thesis/RESULTS', type=str,
                    help='directory where SR images will be saved')

if __name__ == '__main__':
    arg = parser.parse_args()

    mode = arg.mode
    loss = arg.loss
    dim = arg.LR_input_size
    test_data_dir = arg.testdata_dir
    load_weight_dir = arg.load_weight_dir
    HR_folder = arg.HR_folder
    LR_folder = arg.LR_folder
    save_result_dir = arg.save_result_dir

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    test_data_size = len(os.listdir(os.path.join(test_data_dir, HR_folder)))
    lr_image_size = (dim, dim, 3)

    # load model
    input, output = generator(input_size=lr_image_size)
    generator = Model(input, output)
    generator.summary()

    # compile the model
    opt = Adam(lr=0.001)
    if loss == 'mse':
        generator.compile(optimizer=opt, loss='mse', metrics=[psnr, ssim])
    elif loss == 'perceptual':
        generator.compile(optimizer=opt, loss='perceptual_loss', metrics=[psnr, ssim])
    else:
        print("please select loss function between mse or perceptual")

    # load weight
    generator.load_weights(load_weight_dir)

    # Inference
    testGene = testGenerator(sample_size=1000, test_path=test_data_dir, imagelow_folder=LR_folder)
    results = generator.predict_generator(testGene, steps=test_data_size, verbose=1)

    # Save generated SR images
    save_result(save_result_dir, results, testGene)
