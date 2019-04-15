import os

import numpy as np
import tables
from scipy.ndimage import zoom

from fetal_net.utils.utils import read_img, resize
from .normalize import normalize_data_storage, normalize_data_storage_each


def create_data_file(out_file, n_samples):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_vlarray(hdf5_file.root, 'data', tables.ObjectAtom(), filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_vlarray(hdf5_file.root, 'truth', tables.ObjectAtom(), filters=filters, expectedrows=n_samples)
    mask_storage = hdf5_file.create_vlarray(hdf5_file.root, 'mask', tables.ObjectAtom(), filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, mask_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, mask_storage, truth_dtype=np.uint8, scale=None,
                             preproc=None):
    for set_of_files in image_files:
        images = [read_img(_) for _ in set_of_files]
        subject_data = [image.get_data() for image in images]
        if scale is not None:
            subject_data[0] = zoom(subject_data[0], scale) # for sub_data in subject_data]
            subject_data[1] = zoom(subject_data[1], scale, order=0) # for sub_data in subject_data]
        if preproc is not None:
            subject_data[0] = preproc(subject_data[0])
        print(subject_data[0].shape)
        add_data_to_storage(data_storage, truth_storage, mask_storage, subject_data, truth_dtype)
    return data_storage, truth_storage, mask_storage


def add_data_to_storage(data_storage, truth_storage, mask_storage, subject_data, truth_dtype):
    data_storage.append(np.asarray(subject_data[0]).astype(np.float))
    truth_storage.append(np.asarray(subject_data[1], dtype=truth_dtype))
    if len(subject_data) > 2:
        mask_storage.append(np.asarray(subject_data[2]).astype(np.float))


def write_data_to_file(training_data_files, out_file, truth_dtype=np.uint8,
                       subject_ids=None, normalize='all', scale=None, preproc=None):
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
    n_samples = len(training_data_files)
    try:
        hdf5_file, data_storage, truth_storage, mask_storage = create_data_file(out_file, n_samples=n_samples)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, mask_storage,
                             truth_dtype=truth_dtype, scale=scale, preproc=preproc)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if isinstance(normalize, str):
        _, mean, std = {
            'all': normalize_data_storage,
            'each': normalize_data_storage_each
        }[normalize](data_storage)
    else:
        mean, std = None, None
    hdf5_file.close()
    return out_file, (mean, std)


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
