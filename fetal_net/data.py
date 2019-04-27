import os

import numpy as np
from scipy.ndimage import zoom

from fetal_net.utils.utils import read_img, resize, pickle_dump, pickle_load
from .normalize import normalize_data_storage, normalize_data_storage_each

def write_image_data_to_file(image_files, data_storage, subject_ids, scale=None, preproc=None):
    for subject_id, set_of_files in zip(subject_ids, image_files):
        images = [read_img(_) for _ in set_of_files]
        subject_data = [image.get_data() for image in images]
        if scale is not None:
            subject_data[0] = zoom(subject_data[0], scale) # for sub_data in subject_data]
            subject_data[1] = zoom(subject_data[1], scale, order=0) # for sub_data in subject_data]
        if preproc is not None:
            subject_data[0] = preproc(subject_data[0])
        print(subject_data[0].shape)
        add_data_to_storage(data_storage, subject_id, subject_data)
    return data_storage


def add_data_to_storage(storage_dict, subject_id, subject_data):
    storage_dict[subject_id] = {}
    storage_dict[subject_id]['data'] = np.asarray(subject_data[0]).astype(np.float)
    storage_dict[subject_id]['truth'] = np.asarray(subject_data[1]).astype(np.float)
    if len(subject_data) > 2:
        storage_dict[subject_id]['mask'] = np.asarray(subject_data[2]).astype(np.float)


def write_data_to_file(training_data_files, out_file, subject_ids, normalize='all', scale=None, preproc=None):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it. 
    """
    data_dict = {}

    write_image_data_to_file(training_data_files, data_dict, subject_ids, scale=scale, preproc=preproc)

    if isinstance(normalize, str):
        _, mean, std = {
            'all': normalize_data_storage,
            'each': normalize_data_storage_each
        }[normalize](data_dict)
    else:
        mean, std = None, None

    pickle_dump(data_dict, out_file)

    return out_file, (mean, std)


def open_data_file(filename):
    return pickle_load(filename)
