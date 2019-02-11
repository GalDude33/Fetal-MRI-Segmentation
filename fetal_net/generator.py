import os
import random
import numpy as np
from keras.utils import to_categorical, Sequence

from fetal.utils import AttributeDict as att_dict
from fetal_net.utils.utils import resize
from .augment import augment_data, random_permutation_x_y, get_image
from .utils import pickle_dump, pickle_load
from .utils.patches import get_patch_from_3d_data


class DataFileDummy:
    def __init__(self, file, pad=0):
        self.data = [np.pad(_, pad, 'constant', constant_values=_.min()) for _ in file.root.data]
        self.truth = [np.pad(_, pad, 'constant', constant_values=0) for _ in file.root.truth]

        if len(file.root.mask):
            self.mask = [_ for _ in file.root.mask]
        else:
            self.mask = None

        self.stats = att_dict(
            p1=[np.percentile(_, q=1) for _ in self.data],
            min=[np.min(_) for _ in self.data],
            max=[np.max(_) for _ in self.data],
        )
        self.subject_ids = [_ for _ in file.root.subject_ids]

        self.root = self


def pad_samples(data_file, patch_shape, truth_downsample):
    output_shape = [patch_shape[0] // truth_downsample,
                    patch_shape[1] // truth_downsample,
                    1]
    padding = np.ceil(np.subtract(patch_shape, output_shape) / 2).astype(int)

    data_file.root.data = \
        [np.pad(data, [(_, _) for _ in padding], 'constant', constant_values=data_min)
         for data, data_min in zip(data_file.data, data_file.stats.min)]
    data_file.root.truth = \
        [np.pad(truth, [(_, _) for _ in padding], 'constant', constant_values=0)
         for truth in data_file.truth]

    data_file.root.data = \
        [np.pad(data,
                [(_, _) for _ in np.ceil(np.maximum(np.subtract(patch_shape, data.shape) + 1, 0) / 2).astype(int)],
                'constant', constant_values=data_min)
         for data, data_min in zip(data_file.data, data_file.stats.min)]
    data_file.root.truth = \
        [np.pad(truth,
                [(_, _) for _ in np.ceil(np.maximum(np.subtract(patch_shape, truth.shape) + 1, 0) / 2).astype(int)],
                'constant', constant_values=0)
         for truth in data_file.truth]


def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           test_keys_file,
                                           patch_shape=None, data_split=0.8, overwrite=False, labels=None, augment=None,
                                           validation_batch_size=None, skip_blank_train=True, skip_blank_val=False,
                                           truth_index=-1, truth_size=1, truth_downsample=None, truth_crop=True,
                                           patches_per_epoch=1,
                                           categorical=True, is3d=False,
                                           prev_truth_index=None, prev_truth_size=None,
                                           drop_easy_patches_train=False, drop_easy_patches_val=False,
                                           samples_pad=3, val_augment=None):
    """
    Creates the training and validation generators that can be used when training the model.
    :param prev_truth_inedx:
    :param categorical:
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

    data_file = DataFileDummy(data_file, samples_pad)

    pad_samples(data_file, patch_shape, truth_downsample or 1)

    training_list, validation_list, test_list = get_validation_split(data_file,
                                                                     data_split=data_split,
                                                                     overwrite=overwrite,
                                                                     training_file=training_keys_file,
                                                                     validation_file=validation_keys_file,
                                                                     test_file=test_keys_file)

    print("Training: {}".format([data_file.subject_ids[_].decode() for _ in training_list]))
    print("Validation: {}".format([data_file.subject_ids[_].decode() for _ in validation_list]))
    print("Test: {}".format([data_file.subject_ids[_].decode() for _ in test_list]))

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = patches_per_epoch // batch_size
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = patches_per_epoch // validation_batch_size
    print("Number of validation steps: ", num_validation_steps)

    training_generator = \
        data_generator(data_file=data_file, index_list=training_list, batch_size=batch_size,
                       augment=augment,
                       n_labels=n_labels, labels=labels, patch_shape=patch_shape,
                       skip_blank=skip_blank_train,
                       truth_index=truth_index, truth_size=truth_size,
                       truth_downsample=truth_downsample, truth_crop=truth_crop,
                       categorical=categorical, is3d=is3d,
                       prev_truth_index=prev_truth_index, prev_truth_size=prev_truth_size,
                       drop_easy_patches=drop_easy_patches_train)
    validation_generator = \
        data_generator(data_file=data_file, index_list=validation_list, batch_size=validation_batch_size,
                       augment=val_augment,
                       n_labels=n_labels, labels=labels, patch_shape=patch_shape,
                       skip_blank=skip_blank_val,
                       truth_index=truth_index, truth_size=truth_size,
                       truth_downsample=truth_downsample, truth_crop=truth_crop,
                       categorical=categorical, is3d=is3d,
                       prev_truth_index=prev_truth_index, prev_truth_size=prev_truth_size,
                       drop_easy_patches=drop_easy_patches_val)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1


def get_validation_split(data_file, training_file, validation_file, test_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param o
    verwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = len(data_file.root.data)
        sample_list = list(range(nb_samples))
        random.shuffle(sample_list)
        test_list = [sample_list.pop()]
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        pickle_dump(test_list, test_file)
        return training_list, validation_list, test_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file), pickle_load(test_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        random.shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def random_list_generator(index_list):
    while True:
        np.random.seed()
        yield from random.sample(index_list, len(index_list))


def list_generator(index_list):
    while True:
        yield from index_list


class FetalSequence(Sequence):

    def __init__(self, epoch_size, **kargs):
        self.kargs = kargs
        self.generator = data_generator(**kargs)
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        next(self.generator)

    def reset(self):
        self.generator = data_generator(**self.kargs)


def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=None, patch_shape=None,
                   shuffle_index_list=True, skip_blank=True, truth_index=-1, truth_size=1, truth_downsample=None,
                   truth_crop=True, categorical=True, prev_truth_index=None, prev_truth_size=None,
                   drop_easy_patches=False, is3d=False):
    index_generator = random_list_generator(index_list) if shuffle_index_list else list_generator(index_list)
    while True:
        x_list = list()
        y_list = list()
        mask_list = list()

        while len(x_list) < batch_size:
            index = next(index_generator)
            add_data(x_list, y_list, mask_list, data_file, index, augment=augment,
                     patch_shape=patch_shape, skip_blank=skip_blank,
                     truth_index=truth_index, truth_size=truth_size, truth_downsample=truth_downsample,
                     truth_crop=truth_crop, prev_truth_index=prev_truth_index,
                     prev_truth_size=prev_truth_size, drop_easy_patches=drop_easy_patches)
        yield convert_data(x_list, y_list, mask_list, n_labels=n_labels, labels=labels, categorical=categorical,
                           is3d=is3d)


def add_data(x_list, y_list, mask_list, data_file, index, truth_index, truth_size=1, augment=None, patch_shape=None,
             skip_blank=True,
             truth_downsample=None, truth_crop=True, prev_truth_index=None, prev_truth_size=None,
             drop_easy_patches=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param prev_truth_index:
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
    data, truth, mask = get_data_from_file(data_file, index, patch_shape=None)

    patch_corner = [
        np.random.randint(low=low, high=high)
        for low, high in zip((0, 0, 0), truth.shape - np.array(patch_shape))  # - np.array(patch_shape) // 2)
    ]
    if augment is not None:
        data_range = [(start, start + size) for start, size in zip(patch_corner, patch_shape)]

        truth_range = data_range[:2] + [(patch_corner[2] + truth_index,
                                         patch_corner[2] + truth_index + truth_size)]

        if prev_truth_index is not None:
            prev_truth_range = data_range[:2] + [(patch_corner[2] + prev_truth_index,
                                                  patch_corner[2] + prev_truth_index + prev_truth_size)]
        else:
            prev_truth_range = None

        data, truth, prev_truth, mask = \
            augment_data(data, truth,
                         data_min=data_file.stats.min[index],
                         data_max=data_file.stats.max[index],
                         mask=mask,
                         scale_deviation=augment.get('scale', None),
                         iso_scale_deviation=augment.get('iso_scale', None),
                         rotate_deviation=augment.get('rotate', None),
                         translate_deviation=augment.get('translate', None),
                         flip=augment.get('flip', None),
                         contrast_deviation=augment.get('contrast', None),
                         piecewise_affine=augment.get('piecewise_affine', None),
                         elastic_transform=augment.get('elastic_transform', None),
                         intensity_multiplication_range=augment.get('intensity_multiplication', None),
                         poisson_noise=augment.get("poisson_noise", None),
                         gaussian_noise=augment.get("gaussian_noise", None),
                         speckle_noise=augment.get("speckle_noise", None),
                         gaussian_filter=augment.get("gaussian_filter", None),
                         coarse_dropout=augment.get("coarse_dropout", None),
                         data_range=data_range, truth_range=truth_range,
                         prev_truth_range=prev_truth_range)
    else:
        data, truth, prev_truth, mask = \
            extract_patch(data, patch_corner, patch_shape, truth, mask,
                          truth_index=truth_index, truth_size=truth_size,
                          prev_truth_index=prev_truth_index, prev_truth_size=prev_truth_size)

    if prev_truth is not None:
        data = np.concatenate([data, prev_truth], axis=-1)

    if drop_easy_patches:
        truth_mean = np.mean(truth[16:-16, 16:-16, :])
        if 1 - np.abs(truth_mean - 0.5) < np.random.random():
            return

    if truth_downsample is not None and truth_downsample > 1:
        truth_shape = patch_shape[:-1] + (1,)
        new_shape = np.array(truth_shape)
        new_shape[:-1] = new_shape[:-1] // truth_downsample
        if truth_crop:
            truth = get_patch_from_3d_data(truth,
                                           new_shape,
                                           list(np.subtract(truth_shape[:2], new_shape[:2]) // 2) + [1])
        else:
            truth = resize(get_image(truth), new_shape=new_shape).get_data()

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)
        if mask is not None:
            mask_list.append(mask)


def extract_patch(data, patch_corner, patch_shape, truth, mask, truth_index, truth_size, prev_truth_index=None,
                  prev_truth_size=1):
    data = get_patch_from_3d_data(data, patch_shape, patch_corner)
    real_truth = get_patch_from_3d_data(truth,
                                        patch_shape[:-1] + (truth_size,),
                                        patch_corner + np.array((0, 0, truth_index)))
    if mask is not None:
        mask = get_patch_from_3d_data(mask,
                                      patch_shape[:-1] + (truth_size,),
                                      patch_corner + np.array((0, 0, truth_index)))
    if prev_truth_index is not None:
        prev_truth = get_patch_from_3d_data(truth,
                                            patch_shape[:-1] + (prev_truth_size,),
                                            patch_corner + np.array((0, 0, prev_truth_index)))
    else:
        prev_truth = None

    return data, real_truth, prev_truth, mask


def extract_random_patch(data, patch_shape, truth, mask, truth_index, prev_truth_index):
    # cut relevant patch
    patch_corner = [
        np.random.randint(low=low, high=high)
        for low, high in zip((0, 0, 0),  # -np.array(patch_shape) // 2,
                             truth.shape - np.array(patch_shape))  # - np.array(patch_shape) // 2)
    ]
    return extract_patch(data, patch_corner, patch_shape, truth, mask, truth_index, prev_truth_index)


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth, mask = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
        if mask is not None:
            z = get_patch_from_3d_data(mask, patch_shape, patch_index)
        else:
            z = None
    else:
        if data_file.root.mask is not None:
            z = data_file.root.mask[index]
        else:
            z = None
        x, y = data_file.root.data[index], data_file.root.truth[index]
    return x, y, z


def convert_data(x_list, y_list, mask_list, n_labels=1, labels=None, categorical=True, is3d=False):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    masks = np.asarray(mask_list)
    # if n_labels == 1:
    #     y[y > 0] = 1
    # elif n_labels > 1:
    #     y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)

    inputs = []
    if categorical:
        y = to_categorical(y, 2)
    if is3d:
        x = np.expand_dims(x, 1)
        y = np.expand_dims(y, 1)
        masks = np.expand_dims(mask_list, 1)

    inputs = x
    if len(masks) > 0:
        inputs = [x, masks]

    return inputs, y


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
