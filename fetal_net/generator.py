import os
import copy
from random import shuffle
import itertools

import numpy as np
from keras.utils import to_categorical

from fetal_net.utils.utils import resize, read_img
from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y, get_image


def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           patch_shape=None, data_split=0.8, overwrite=False, labels=None, augment=None,
                                           validation_batch_size=None, skip_blank=True, truth_index=-1,
                                           truth_downsample=None):
    """
    Creates the training and validation generators that can be used when training the model.
    :param truth_downsample:
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If not None, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)

    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        patch_shape=patch_shape,
                                        skip_blank=skip_blank,
                                        truth_index=truth_index,
                                        truth_downsample=truth_downsample)
    validation_generator = data_generator(data_file, validation_list, batch_size=validation_batch_size,
                                          n_labels=n_labels, labels=labels, patch_shape=patch_shape,
                                          skip_blank=skip_blank, truth_index=truth_index,
                                          truth_downsample=truth_downsample)

    patches_per_img_per_batch = 10
    # Set the number of training and testing samples per epoch correctly
    num_training_steps = patches_per_img_per_batch * get_number_of_steps(len(training_list), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = patches_per_img_per_batch * get_number_of_steps(len(validation_list), validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=None, patch_shape=None,
                   shuffle_index_list=True, skip_blank=True, truth_index=-1, truth_downsample=None):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()

        index_list = copy.copy(orig_index_list)
        if shuffle_index_list:
            shuffle(index_list)

        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, augment=augment,
                     patch_shape=patch_shape, skip_blank=skip_blank,
                     truth_index=truth_index, truth_downsample=truth_downsample)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()


def add_data(x_list, y_list, data_file, index, truth_index,
             augment=None, patch_shape=None, skip_blank=True, truth_downsample=None):
    """
    Adds data from the data file to the given lists of feature and target data
    :param truth_downsample:
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if not None, data will be augmented according to the augmentation parameters
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=None)
    if augment is not None:
        data, truth = augment_data(data, truth,
                                   flip=augment['flip'],
                                   scale_deviation=augment['scale'],
                                   translate_deviation=augment['translate'],
                                   rotate_deviation=augment['rotate'])

        if augment["permute"] is not None and augment["permute"]:
            if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
                raise ValueError(
                    "To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                    "the same length.")
            data, truth = random_permutation_x_y(data, truth)

    # cut relevant patch
    patch_corner = [
        np.random.randint(low=low, high=high)
        for low, high in zip(-np.array(patch_shape) // 2,
                             truth.shape - np.array(patch_shape) // 2)
    ]
    data = get_patch_from_3d_data(data, patch_shape, patch_corner)

    truth_shape = patch_shape[:-1]+(1,)
    truth = get_patch_from_3d_data(truth,
                                   truth_shape,
                                   patch_corner + np.array((0, 0, truth_index)))
    if truth_downsample is not None and truth_downsample > 1:
        new_shape = np.array(truth_shape)
        new_shape[:-1] = new_shape[:-1] / truth_downsample
        truth = resize(get_image(truth), new_shape=new_shape).get_data()

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index]
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    # if n_labels == 1:
    #     y[y > 0] = 1
    # elif n_labels > 1:
    #     y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    y = to_categorical(y, 2)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y
