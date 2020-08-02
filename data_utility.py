from __future__ import print_function

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt


def sample_dataset(size, data_path, folder, image_color_mode="rgb", seed=1):

    '''
    Using ImageDataGenerator to create sample data
    :param size: how many image data do you want to sample
    :param data_path: directory for image data
    '''

    imagesamp_datagen = ImageDataGenerator(rescale=1.0/255)

    imagesamp_generator = imagesamp_datagen.flow_from_directory(
        data_path,
        classes=[folder],
        batch_size=size,
        shuffle=True, # because we want to randomly choose 'size' of data.
        class_mode=None, # this means our generator will only yield batches of data, no labels
        color_mode=image_color_mode,
        seed=seed
    )

    # Note that you have to use next() so the function can return the 'data'
    # If you use return instead, then you are returning the generator object
    return next(imagesamp_generator)


def trainGenerator(batch_size, sample_size, train_path, imagelow_folder, imagehigh_folder, image_color_mode="rgb",
                   lr_target_size=(88, 88), hr_target_size=(352, 352), seed=1):
    '''
    can generate lowres image and highres image at the same time
    use the same seed for both datagens to ensure the transformations are the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    # creating the object of ImageDataGenerator
    imagelow_datagen = ImageDataGenerator(rescale=1.0/255, featurewise_center=True, featurewise_std_normalization=True)
    imagehigh_datagen = ImageDataGenerator(rescale=1.0/255, featurewise_center=True, featurewise_std_normalization=True)

    sampled_low = sample_dataset(size=sample_size, data_path=train_path, folder=imagelow_folder)
    sampled_high = sample_dataset(size=sample_size, data_path=train_path, folder=imagehigh_folder)

    imagelow_datagen.fit(sampled_low)
    imagehigh_datagen.fit(sampled_high)

    # passing the data files and settings
    imagelow_generator = imagelow_datagen.flow_from_directory(
        train_path,
        classes = [imagelow_folder],  # class subdirectories
        class_mode = None, # this means our generator will only yield batches of data, no labels
        color_mode = image_color_mode,
        target_size = lr_target_size,
        batch_size = batch_size,
        seed = seed)

    imagehigh_generator = imagehigh_datagen.flow_from_directory(
        train_path,
        classes = [imagehigh_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = hr_target_size,
        batch_size = batch_size,
        seed = seed)

    train_generator = zip(imagelow_generator, imagehigh_generator)
    for (imglow, imghigh) in train_generator:
        imglow, imghigh = imglow, imghigh
        yield imglow, imghigh


def valGenerator(batch_size, sample_size, train_path, imagelow_folder, imagehigh_folder, image_color_mode="rgb",
                 lr_target_size=(88, 88), hr_target_size=(352, 352), seed=1):
    '''
    can generate lowres image and highres image at the same time
    use the same seed for both datagens to ensure the transformations are the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    imagelow_datagen = ImageDataGenerator(rescale=1.0/255, featurewise_center=True, featurewise_std_normalization=True)
    imagehigh_datagen = ImageDataGenerator(rescale=1.0/255, featurewise_center=True, featurewise_std_normalization=True)

    sampled_low = sample_dataset(size=sample_size, data_path=train_path, folder=imagelow_folder)
    sampled_high = sample_dataset(size=sample_size, data_path=train_path, folder=imagehigh_folder)

    imagelow_datagen.fit(sampled_low)
    imagehigh_datagen.fit(sampled_high)

    imagelow_generator = imagelow_datagen.flow_from_directory(
        train_path,
        classes=[imagelow_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=lr_target_size,
        batch_size=batch_size,
        seed=seed)

    train_path_high = os.path.join(train_path,imagehigh_folder)
    imagehigh_generator = imagehigh_datagen.flow_from_directory(
        train_path,
        classes=[imagehigh_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=hr_target_size,
        batch_size=batch_size,
        seed=seed)

    train_generator = zip(imagelow_generator, imagehigh_generator)
    for (imglow, imghigh) in train_generator:
        imglow, imghigh = imglow, imghigh
        yield imglow, imghigh


def testGenerator(sample_size, test_path, imagelow_folder, lr_target_size=(88, 88), image_color_mode="rgb", as_gray = True):

    test_datagen = ImageDataGenerator(rescale=1.0/255, featurewise_center=True, featurewise_std_normalization=True)

    sampled = sample_dataset(size=sample_size, data_path=test_path, folder=imagelow_folder)

    test_datagen.fit(sampled)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        classes=[imagelow_folder],
        batch_size=1, # is this number changes, change also 'steps' in the call to model.predict_generator
        color_mode=image_color_mode,
        target_size=lr_target_size,
        shuffle=False,
        class_mode=None)

    # return test_generator
    return test_generator


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def yield_generator(test_generator):
    for img in test_generator:
        yield img

def save_result(save_path, npyfile, test_generator):
    """save file with corresponding filename"""
    for i,item in enumerate(npyfile):
        item = item / 255.0 # Back to range [0,1] from [0, 255]
        img = rgb2gray(item)
        filepath = test_generator.filenames[i] # image/PXXX_etc.png
        name = os.path.split(filepath)[-1]
        plt.imsave(os.path.join(save_path,name), img, cmap=plt.get_cmap('gray'))


def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,1.0)


def ssim(y_true, y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)