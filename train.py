import argparse
from math import ceil

from network import *
from data_utility import *
from perceptual_loss import *

from keras.models import Model
from keras.layers import Lambda

parser = argparse.ArgumentParser(description='Train 2D super-resolution ResNet')
parser.add_argument('--init_epoch', default=0, type=int, help="initial epoch")
parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--mode', default='PS', type=str, help='upscaling method, choose between PS or NN')
parser.add_argument('--loss', default='mse', type=str, help='loss function, choose between mse or perceptual')
parser.add_argument('--upscale_factor', default=4, type=int, help='upscaling factor')
parser.add_argument('--LR_input_size', default=88, type=int, help='width or height of input dim')
parser.add_argument('--train_data_dir', default='/media/saewon/Data/Saewon_thesis/Dataset/train', type=str,
                    help='directory where LR and HR volumes are saved for training')
parser.add_argument('--val_data_dir', default='/media/saewon/Data/Saewon_thesis/Dataset/validation', type=str,
                    help='directory where LR and HR volumes are saved for validation')
parser.add_argument('--load_weight_dir', default=None,
                    type=str,
                    help='directory for loading the weights from, if you are training the network from scratch, set to None')
parser.add_argument('--save_weight_dir',
                    default='./checkpoints/tempcheckpoints',
                    type=str, help='directory where training weights will be saved')
parser.add_argument('--log_dir', default='/home/saewon/Documents/tensorboard_outputs/SRResNet/super_res',
                    type=str, help='log directory for tensorboard')
parser.add_argument('--HR_folder', default='bssfp', type=str, help='folder where HR volumes are saved')
parser.add_argument('--LR_folder', default='bssfp_lowres', type=str, help='folder where LR volumes are saved')

if __name__ == '__main__':
    arg = parser.parse_args()

    init_epoch = arg.init_epoch
    num_epochs = arg.num_epochs
    batch_size = arg.batch_size
    mode = arg.mode
    loss = arg.loss
    upscale_factor = arg.upscale_factor
    dim = arg.LR_input_size
    train_data_dir = arg.train_data_dir
    val_data_dir = arg.val_data_dir
    load_weight_dir = arg.load_weight_dir
    save_weight_dir = arg.save_weight_dir
    log_dir = arg.log_dir
    HR_folder = arg.HR_folder
    LR_folder = arg.LR_folder

    if not os.path.exists(save_weight_dir):
        os.makedirs(save_weight_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sample_size = 1000  # set the size of data which will be sampled for pre-processing
    lr_image_size = (dim, dim, 3)
    hr_image_size = (dim * upscale_factor, dim * upscale_factor, 3)
    train_data_size = len(os.listdir(os.path.join(train_data_dir, HR_folder)))
    val_data_size = len(os.listdir(os.path.join(val_data_dir, HR_folder)))

    steps_per_epoch = ceil(train_data_size / batch_size)
    val_steps = ceil(val_data_size / batch_size)

    # load data generator
    train_loader = trainGenerator(batch_size, sample_size, train_data_dir, LR_folder, HR_folder,
                                  lr_target_size=(dim, dim), hr_target_size=(dim*upscale_factor, dim*upscale_factor))
    val_loader = valGenerator(batch_size, sample_size, val_data_dir, LR_folder, HR_folder,
                              lr_target_size=(dim, dim), hr_target_size=(dim*upscale_factor, dim*upscale_factor))

    # create VGG model
    vgg_inp, vgg_outp = get_VGG19(input_size=hr_image_size)
    vgg_content = Model(vgg_inp, vgg_outp)
    vgg_content.summary()


    # perceptual loss
    def perceptual_loss(y_true, y_pred):
        # mse=K.losses.mean_squared_error(y_true,y_pred)

        # Lambda creates a layer from a function. This makes this preprocessing step a layer
        # This subtracts the ImageNet group mean to all images. Also BGR -> RGB with x[:, :, ::-1]
        rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
        preproc_layer = Lambda(preproc)

        y_t = vgg_content(preproc_layer(y_true))
        y_p = vgg_content(preproc_layer(y_pred))

        loss = keras.losses.mean_squared_error(y_t, y_p)
        return loss


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

    # load weight and continue from previous training?
    if load_weight_dir is not None:
        generator.load_weights(load_weight_dir)

    # set tensorboard details
    filename = "SRResNet-{epoch:02d}-{loss:.2f}.hdf5"
    filepath = os.path.join(save_weight_dir, filename)

    model_checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False,
                                       mode='auto')

    tensor_board_callback = TensorBoard(log_dir=log_dir,
                                        histogram_freq=0,
                                        # batch_size=batch_size, # Deprecated in Tensorflow 2.0
                                        write_graph=True,
                                        write_grads=False,  # True gives error!
                                        write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                        embeddings_data=None, update_freq='epoch'
                                        )

    progbar_logger = ProgbarLogger(count_mode='steps')

    callbacks = [model_checkpoint, tensor_board_callback, progbar_logger]

    history = generator.fit_generator(generator=train_loader,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=num_epochs,
                                      callbacks=callbacks,
                                      validation_data=val_loader,
                                      validation_steps=val_steps,
                                      initial_epoch=init_epoch,
                                      verbose=1  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                      )




