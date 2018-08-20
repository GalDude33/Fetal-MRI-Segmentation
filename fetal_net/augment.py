import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools

from fetal_net.utils.utils import get_image, interpolate_affine_range


def scale_image(affine, scale_factor):
    scale_factor = np.diag(list(scale_factor)+[1])
    new_affine = scale_factor.dot(affine)
    return new_affine


def translate_image(affine, translate_factor):
    translate_factor = np.asarray(translate_factor)
    new_affine = np.copy(affine)
    new_affine[0:3, 3] = new_affine[0:3, 3] + translate_factor
    return new_affine


def rotate_image_axis(affine, rotate_factor, axis):
    return {
        0: rotate_image_x,
        1: rotate_image_y,
        2: rotate_image_z
    }[axis](affine, rotate_factor)


def rotate_image_x(affine, rotate_factor):
    sin_gamma = np.sin(rotate_factor)
    cos_gamma = np.cos(rotate_factor)
    rotation_affine = np.array([[1, 0, 0, 0],
                                [0, cos_gamma, -sin_gamma, 0],
                                [0, sin_gamma, cos_gamma, 0],
                                [0, 0, 0, 1]])
    new_affine = rotation_affine.dot(affine)
    return new_affine


def rotate_image_y(affine, rotate_factor):
    sin_gamma = np.sin(rotate_factor)
    cos_gamma = np.cos(rotate_factor)
    rotation_affine = np.array([[cos_gamma, 0, sin_gamma, 0],
                                [0, 1, 0, 0],
                                [-sin_gamma, 0, cos_gamma, 0],
                                [0, 0, 0, 1]])
    new_affine = rotation_affine.dot(affine)
    return new_affine


def rotate_image_z(affine, rotate_factor):
    sin_gamma = np.sin(rotate_factor)
    cos_gamma = np.cos(rotate_factor)
    rotation_affine = np.array([[1, 0, 0, 0],
                                [0, cos_gamma, -sin_gamma, 0],
                                [0, sin_gamma, cos_gamma, 0],
                                [0, 0, 0, 1]])
    new_affine = rotation_affine.dot(affine)
    return new_affine


def rotate_image(affine, rotate_angles):
    new_affine = np.copy(affine)

    # apply rotations
    for i, rotate_angle in enumerate(rotate_angles):
        if rotate_angle > 0:
            new_affine = rotate_image_axis(new_affine, rotate_angle, axis=i)

    return new_affine


def flip_image(affine, axis):
    new_affine = np.copy(affine)

    for ax in axis:
        new_affine = rotate_image_axis(new_affine, np.deg2rad(180), axis=ax)

    return new_affine


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_translate_factor(n_dim=3, mean=0, std=0.10):
    return np.random.normal(mean, std, n_dim)


def random_rotation_angle(n_dim=3, mean=0, std=5):
    return np.random.normal(mean, std, n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(data, affine, flip_axis=None, scale_factor=None, rotate_factor=None):
    # translate center of image to 0,0,0
    center_offset = np.array(data.shape)/2
    affine = translate_image(affine, -center_offset)

    if flip_axis is not None:
        affine = flip_image(affine, flip_axis)
    if scale_factor is not None:
        affine = scale_image(affine, scale_factor)
    if rotate_factor is not None:
        affine = rotate_image(affine, rotate_factor)

    # translate image back to original coordinates
    affine = translate_image(affine, +center_offset)

    return data, affine


def random_flip_dimensions(n_dim, flip_factor):
    return np.arange(n_dim)[
        [flip_rate > random.random()
         for flip_rate in flip_factor]
    ]


def augment_data(data, truth, data_min, scale_deviation=None, rotate_deviation=None,
                 flip=True, data_range=None, truth_range=None):
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = None
    if rotate_deviation:
        rotate_factor = random_rotation_angle(n_dim, std=rotate_deviation)
        rotate_factor = np.deg2rad(rotate_factor)
    else:
        rotate_factor = None
    if flip is not None and flip:
        flip_axis = random_flip_dimensions(n_dim, flip)
    else:
        flip_axis = None

    image, affine = data, np.eye(4)
    distorted_data, distorted_affine = distort_image(image, affine,
                                                     flip_axis=flip_axis,
                                                     scale_factor=scale_factor,
                                                     rotate_factor=rotate_factor)
    if data_range is None:
        data = resample_to_img(get_image(distorted_data, distorted_affine), image, interpolation="continuous",
                               copy=False, clip=True).get_fdata()
    else:

        data = interpolate_affine_range(distorted_data, distorted_affine, data_range, order=1, mode='constant',
                                        cval=data_min)

    truth_image, truth_affine = truth, np.eye(4)
    distorted_truth_data, distorted_truth_affine = distort_image(truth_image, truth_affine,
                                                                 flip_axis=flip_axis,
                                                                 scale_factor=scale_factor,
                                                                 rotate_factor=rotate_factor)
    if truth_range is None:
        truth_data = resample_to_img(get_image(distorted_truth_data, distorted_truth_affine), truth_image,
                                     interpolation="nearest", copy=False,
                                     clip=True).get_data()
    else:
        truth_data = interpolate_affine_range(distorted_truth_data, distorted_truth_affine, truth_range, order=0,
                                              mode='constant', cval=0)
    return data, truth_data


def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.

    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.

    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 2))
    # if rotate_z != 0:
    #     data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    # if transpose:
    #    for i in range(data.shape[0]):
    #        data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)


def reverse_permute_data(data, key):
    key = reverse_permutation_key(key)
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    # if transpose:
    #    for i in range(data.shape[0]):
    #        data[i] = data[i].T
    if flip_z:
        data = data[:, :, :, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_x:
        data = data[:, ::-1]
    # if rotate_z != 0:
    #    data = np.rot90(data, rotate_z, axes=(2, 3))
    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 2))
    return data


def reverse_permutation_key(key):
    rotation = tuple([-rotate for rotate in key[0]])
    return rotation, key[1], key[2], key[3], key[4]
