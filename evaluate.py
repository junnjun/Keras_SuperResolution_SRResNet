import argparse

from network import *
from data_utility import *

from keras.models import Model

parser = argparse.ArgumentParser(description='Inference 2D super-resolution ResNet')
parser.add_argument('--batch_size', default=16, type=int, help='batch size that is used for training the network')
parser.add_argument('--upscale_factor', default=4, type=int, help='upscaling factor')
parser.add_argument('--mode', default='PS', type=str, help='upscaling method, choose between PS or NN')
parser.add_argument('--loss', default='mse', type=str, help='loss function, choose between mse or perceptual')
parser.add_argument('--LR_input_size', default=88, type=int, help='width or height of input dim')
parser.add_argument('--test_data_dir', default='/media/saewon/Data/Saewon_thesis/Dataset/validation', type=str,
                    help='directory where LR and HR volumes are saved for training')
parser.add_argument('--load_weight_dir', default= './checkpoints/tempcheckpoints/perceptual_SR-11-0.04.hdf5',
                    type=str,
                    help='which weight to load?')
parser.add_argument('--HR_folder', default='bssfp', type=str, help='folder where HR volumes are saved')
parser.add_argument('--LR_folder', default='bssfp_lowres', type=str, help='folder where LR volumes are saved')
parser.add_argument('--save_result_dir', default='/media/saewon/Data/Saewon_thesis/RESULTS/SRResNet/results', type=str,
                    help='directory where SR images will be saved')

if __name__ == '__main__':
    arg = parser.parse_args()

    batch_size = arg.batch_size
    upscale_factor = arg.upscale_factor
    mode = arg.mode
    loss = arg.loss
    dim = arg.LR_input_size
    test_data_dir = arg.test_data_dir
    load_weight_dir = arg.load_weight_dir
    HR_folder = arg.HR_folder
    LR_folder = arg.LR_folder
    save_result_dir = arg.save_result_dir

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    test_data_size = len(os.listdir(os.path.join(test_data_dir, HR_folder)))
    lr_image_size = (dim, dim, 3)
    hr_image_size = (dim * upscale_factor, dim * upscale_factor, 3)

    # load model
    input, output = generator(batch_size=batch_size, input_size=lr_image_size, upscale=upscale_factor, mode=mode)
    generator = Model(input, output)
    generator.summary()

    # create VGG model
    vgg_inp, vgg_outp = get_VGG16(input_size=hr_image_size)
    vgg_content = Model(vgg_inp, vgg_outp)
    vgg_content.summary()

    def perceptual_loss(y_true, y_pred):
        # mse=K.losses.mean_squared_error(y_true,y_pred)

        # Lambda creates a layer from a function. This makes this preprocessing step a layer
        # This subtracts the ImageNet group mean to all images. Also BGR -> RGB with x[:, :, ::-1]
        rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
        preproc_layer = Lambda(preproc)

        y_t = vgg_content(preproc_layer(y_true))
        y_p = vgg_content(preproc_layer(y_pred))

        loss = tf.keras.losses.mean_squared_error(y_t, y_p)
        return loss

    # compile the model
    opt = Adam(lr=0.001)
    if loss == 'mse':
        generator.compile(optimizer=opt, loss='mse', metrics=[psnr, ssim])
    elif loss == 'perceptual':
        generator.compile(optimizer=opt, loss=perceptual_loss, metrics=[psnr, ssim])
    else:
        print("please select loss function between mse or perceptual")

    # load weight
    generator.load_weights(load_weight_dir)

    # Inference
    testGene = testGenerator(sample_size=1000, test_path=test_data_dir, imagelow_folder=LR_folder)
    enumerator = yield_generator(test_generator=testGene)
    results = generator.predict_generator(enumerator, steps=test_data_size, verbose=1)

    # Save generated SR images
    save_result(save_result_dir, results, test_generator=testGene)
