import numpy as np
from nilearn.image import resample_to_img
import random
import itertools

from skimage.exposure import exposure
from skimage.filters import gaussian
from skimage.util import random_noise

from fetal_net.utils.utils import get_image, interpolate_affine_range, MinMaxScaler
from imgaug import augmenters as iaa


def scale_image(affine, scale_factor):
    scale_factor = np.diag(list(scale_factor) + [1])
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
    rotation_affine = np.array([[cos_gamma, -sin_gamma, 0, 0],
                                [sin_gamma, cos_gamma, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    new_affine = rotation_affine.dot(affine)
    return new_affine


def rotate_image(affine, rotate_angles):
    new_affine = np.copy(affine)

    # apply rotations
    for i, rotate_angle in enumerate(rotate_angles):
        if rotate_angle != 0:
            new_affine = rotate_image_axis(new_affine, rotate_angle, axis=i)

    return new_affine


def flip_image(affine, axis):
    new_affine = np.copy(affine)

    for ax in axis:
        new_affine = rotate_image_axis(new_affine, np.deg2rad(180), axis=ax)

    return new_affine


def shot_noise(data):
    mm_scaler = MinMaxScaler((0, 1))
    data = mm_scaler.fit_transform(data)
    # TODO: remove hardcoded quantization number :(
    data = np.floor(data * 1023) / 1023  # quantization of the data is needed before poisson noise
    # TODO: check if clip=True really needed
    new_data = random_noise(data, mode='poisson', clip=True)
    return mm_scaler.inverse_transform(new_data)


def add_gaussian_noise(data, sigma):
    mm_scaler = MinMaxScaler((0, 1))
    data = mm_scaler.fit_transform(data)
    new_data = random_noise(data, mode='gaussian', clip=True, var=sigma**2)
    return mm_scaler.inverse_transform(new_data)


def add_speckle_noise(data, sigma):
    mm_scaler = MinMaxScaler((0, 1))
    data = mm_scaler.fit_transform(data)
    new_data = random_noise(data, mode='speckle', clip=True, var=sigma**2)
    return mm_scaler.inverse_transform(new_data)


def apply_gaussian_filter(data, sigma):
    return gaussian(data, sigma=sigma)


def apply_coarse_dropout(data, rate, size_percent, per_channel=True):
    mm_scaler = MinMaxScaler((0, 255))
    data = mm_scaler.fit_transform(data)
    new_data = iaa.CoarseDropout(p=rate, size_percent=size_percent, per_channel=per_channel).augment_image(data)
    return mm_scaler.inverse_transform(new_data)


def contrast_augment(data, min_per, max_per):
    # in_range = (np.percentile(data, q=min_per), np.percentile(data, q=max_per))
    in_range = (min_per, max_per)
    return exposure.rescale_intensity(data, in_range=in_range, out_range='image')


def apply_piecewise_affine(data, truth, prev_truth, mask, scale):
    rs = np.random.RandomState()
    vol_pa_transform = iaa.PiecewiseAffine(scale, nb_cols=2, nb_rows=2, order=1, random_state=rs, deterministic=True)
    truth_pa_transform = iaa.PiecewiseAffine(scale, nb_cols=2, nb_rows=2, order=0, random_state=rs, deterministic=True)
    data = vol_pa_transform.augment_image(data)
    truth = truth_pa_transform.augment_image(truth)

    if prev_truth is not None:
        prev_truth_pa_transform = iaa.PiecewiseAffine(scale, nb_cols=2, nb_rows=2, order=0, random_state=rs,
                                                      deterministic=True)
        prev_truth = prev_truth_pa_transform.augment_image(prev_truth)

    if mask is not None:
        mask_pa_transform = iaa.PiecewiseAffine(scale, nb_cols=2, nb_rows=2, order=0, random_state=rs,
                                                deterministic=True)
        mask = mask_pa_transform.augment_image(mask)

    return data, truth, prev_truth, mask


def apply_elastic_transform(data, truth, prev_truth, mask, alpha, sigma):
    rs = np.random.RandomState()
    vol_et_transform = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, order=1, random_state=rs, deterministic=True,
                                                 mode="nearest")
    truth_et_transform = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, order=0, random_state=rs,
                                                   deterministic=True, mode="nearest")

    data = vol_et_transform.augment_image(data)
    truth = truth_et_transform.augment_image(truth)

    if prev_truth is not None:
        prev_truth_et_transform = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, order=0, random_state=rs,
                                                            deterministic=True, mode="nearest")
        prev_truth = prev_truth_et_transform.augment_image(prev_truth)

    if mask is not None:
        mask_et_transform = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, order=0, random_state=rs,
                                                      deterministic=True, mode="nearest")
        mask = mask_et_transform.augment_image(mask)

    return data, truth, prev_truth, mask


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_translate_factor(n_dim=3, min=0, max=7):
    return np.random.uniform(min, max, n_dim)


def random_rotation_angle(n_dim=3, mean=0, std=5):
    return np.random.uniform(low=mean - np.array(std), high=mean + np.array(std), size=n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(data, affine, flip_axis=None, scale_factor=None, rotate_factor=None, translate_factor=None):
    # print('Affine1: ', str(affine))

    # translate center of image to 0,0,0
    center_offset = np.array(data.shape) / 2
    affine = translate_image(affine, -center_offset)
    # print('Affine - center offset: ', str(affine))

    if flip_axis is not None:
        affine = flip_image(affine, flip_axis)
    if scale_factor is not None:
        affine = scale_image(affine, scale_factor)
    if rotate_factor is not None:
        affine = rotate_image(affine, rotate_factor)

    # translate image back to original coordinates
    affine = translate_image(affine, +center_offset)

    if translate_factor is not None:
        affine = translate_image(affine, translate_factor)

    return data, affine


def random_flip_dimensions(n_dim, flip_factor):
    return np.arange(n_dim)[
        [flip_rate > random.random()
         for flip_rate in flip_factor]
    ]


def augment_data(data, truth, data_min, data_max, mask=None, scale_deviation=None, iso_scale_deviation=None,
                 rotate_deviation=None,
                 translate_deviation=None, flip=None, contrast_deviation=None,
                 poisson_noise=None, gaussian_noise=None, speckle_noise=None,
                 piecewise_affine=None, elastic_transform=None, intensity_multiplication_range=None,
                 gaussian_filter=None, coarse_dropout=None, data_range=None, truth_range=None, prev_truth_range=None):
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = [1, 1, 1]
    if iso_scale_deviation:
        iso_scale_factor = np.random.uniform(1, iso_scale_deviation["max"])
        if random_boolean():
            iso_scale_factor = 1 / iso_scale_factor
        scale_factor[0] *= iso_scale_factor
        scale_factor[1] *= iso_scale_factor
    else:
        iso_scale_factor = None
    if rotate_deviation:
        rotate_factor = random_rotation_angle(n_dim, std=rotate_deviation)
        rotate_factor = np.deg2rad(rotate_factor)
    else:
        rotate_factor = None
    if flip is not None and flip:
        flip_axis = random_flip_dimensions(n_dim, flip)
    else:
        flip_axis = None
    if translate_deviation is not None:
        translate_factor = random_translate_factor(n_dim, -np.array(translate_deviation), np.array(translate_deviation))
        translate_factor[-1] = np.floor(translate_factor[-1])  # z-translate should be int
    else:
        translate_factor = None
    if contrast_deviation is not None:
        val_range = data_max - data_min
        contrast_min_val = data_min + contrast_deviation["min_factor"] * np.random.uniform(-1, 1) * val_range
        contrast_max_val = data_max + contrast_deviation["max_factor"] * np.random.uniform(-1, 1) * val_range
    else:
        contrast_min_val, contrast_max_val = None, None
    if poisson_noise is not None:
        apply_poisson_noise = poisson_noise > np.random.random()
    else:
        apply_poisson_noise = False
    if gaussian_noise is not None:
        apply_gaussian_noise = gaussian_noise["prob"] > np.random.random()
    else:
        apply_gaussian_noise = False
    if speckle_noise is not None:
        apply_speckle_noise = speckle_noise["prob"] > np.random.random()
    else:
        apply_speckle_noise = False

    if gaussian_filter is not None and gaussian_filter["prob"] > 0:
        gaussian_sigma = gaussian_filter["max_sigma"] * np.random.random()
        apply_gaussian = gaussian_filter["prob"] > np.random.random()
    else:
        apply_gaussian, gaussian_sigma = False, None
    if piecewise_affine is not None:
        piecewise_affine_scale = np.random.random() * piecewise_affine["scale"]
    else:
        piecewise_affine_scale = 0
    if (elastic_transform is not None) and (elastic_transform["alpha"] > 0):
        elastic_transform_scale = np.random.random() * elastic_transform["alpha"]
    else:
        elastic_transform_scale = 0
    if intensity_multiplication_range is not None:
        a, b = intensity_multiplication_range
        intensity_multiplication = np.random.random() * (b - a) + a
    else:
        intensity_multiplication = 1
    if coarse_dropout is not None:
        coarse_dropout_rate = coarse_dropout['rate']
        coarse_dropout_size = coarse_dropout['size_percent']

    image, affine = data, np.eye(4)
    distorted_data, distorted_affine = distort_image(image, affine,
                                                     flip_axis=flip_axis,
                                                     scale_factor=scale_factor,
                                                     rotate_factor=rotate_factor,
                                                     translate_factor=translate_factor)
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
                                                                 rotate_factor=rotate_factor,
                                                                 translate_factor=translate_factor)
    if truth_range is None:
        truth_data = resample_to_img(get_image(distorted_truth_data, distorted_truth_affine), truth_image,
                                     interpolation="nearest", copy=False,
                                     clip=True).get_data()
    else:
        truth_data = interpolate_affine_range(distorted_truth_data, distorted_truth_affine,
                                              truth_range, order=0, mode='constant', cval=0)

    if prev_truth_range is None:
        prev_truth_data = None
    else:
        prev_truth_data = interpolate_affine_range(distorted_truth_data, distorted_truth_affine,
                                                   prev_truth_range, order=0, mode='constant', cval=0)

    if mask is None:
        mask_data = None
    else:
        mask_image, mask_affine = mask, np.eye(4)
        distorted_mask_data, distorted_mask_affine = distort_image(mask_image, mask_affine,
                                                                   flip_axis=flip_axis,
                                                                   scale_factor=scale_factor,
                                                                   rotate_factor=rotate_factor,
                                                                   translate_factor=translate_factor)
        if truth_range is None:
            mask_data = resample_to_img(get_image(distorted_mask_data, distorted_mask_affine), mask_image,
                                        interpolation="nearest", copy=False,
                                        clip=True).get_data()
        else:
            mask_data = interpolate_affine_range(distorted_mask_data, distorted_mask_affine,
                                                 truth_range, order=0, mode='constant', cval=0)

    if piecewise_affine_scale > 0:
        data, truth_data, prev_truth_data, mask_data = apply_piecewise_affine(data, truth_data,
                                                                              prev_truth_data, mask_data,
                                                                              piecewise_affine_scale)

    if elastic_transform_scale > 0:
        data, truth_data, prev_truth_data, mask_data = apply_elastic_transform(data, truth_data,
                                                                               prev_truth_data, mask_data,
                                                                               elastic_transform_scale,
                                                                               elastic_transform["sigma"])

    if contrast_deviation is not None:
        data = contrast_augment(data, contrast_min_val, contrast_max_val)

    if intensity_multiplication != 1:
        data = data * intensity_multiplication

    if apply_gaussian:
        data = apply_gaussian_filter(data, gaussian_sigma)

    if apply_poisson_noise:
        data = shot_noise(data)

    if apply_speckle_noise:
        data = add_speckle_noise(data, speckle_noise["sigma"])

    if apply_gaussian_noise:
        data = add_gaussian_noise(data, gaussian_noise["sigma"])

    if coarse_dropout is not None:
        data = apply_coarse_dropout(data, rate=coarse_dropout_rate, size_percent=coarse_dropout_size,
                                    per_channel=coarse_dropout["per_channel"])

    return data, truth_data, prev_truth_data, mask_data


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
