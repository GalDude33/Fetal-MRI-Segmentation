import argparse
import json
import os
from itertools import chain, combinations
from pathlib import Path

import numpy as np
from nilearn.image import new_img_like
from scipy import ndimage

import fetal_net.preprocess
from fetal.utils import get_last_model_path

from fetal_net.augment import contrast_augment
from fetal_net.normalize import normalize_data
from fetal_net.postprocess import postprocess_prediction as process_pred
from fetal_net.prediction import patch_wise_prediction
from fetal_net.training import load_old_model
from fetal.preprocess import window_intensities_data

from fetal_net.utils.cut_relevant_areas import find_bounding_box, check_bounding_box
from fetal_net.utils.utils import read_img, get_image
import nibabel as nib


def predict_augment(data, model, overlap_factor, config, num_augments=10):
    data_max = data.max()
    data_min = data.min()

    order = 2
    predictions = []
    for _ in range(num_augments):
        val_range = data_max - data_min
        contrast_min_val = data_min + 0.2 * np.random.uniform(-1, 1) * val_range
        contrast_max_val = data_max + 0.2 * np.random.uniform(-1, 1) * val_range
        curr_data = contrast_augment(data, contrast_min_val, contrast_max_val)

        rotate_factor = np.random.uniform(0, 360)

        to_flip = np.random.choice([True, False])
        if to_flip:
            curr_data = np.flip(curr_data, axis=2)

        curr_data = ndimage.rotate(curr_data, rotate_factor, order=order, reshape=False)
        curr_prediction = \
            ndimage.rotate(patch_wise_prediction(model=model,
                                                 data=curr_data[np.newaxis, ...],
                                                 overlap_factor=overlap_factor,
                                                 patch_shape=config["patch_shape"] + [config["patch_depth"]]).squeeze(),
                           -rotate_factor, order=order, reshape=False)
        if to_flip:
            curr_prediction = np.flip(curr_prediction, axis=2)

        predictions += [curr_prediction]

    return predictions


def predict_flips(data, model, overlap_factor, config):
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))

    def flip_it(data, axes):
        if 0 in axes:
            data = data[::-1, ...]
        if 1 in axes:
            data = data[:, ::-1, ...]
        if 2 in axes:
            data = data[:, :, ::-1, ...]
        return data

    def predict_it(data, axes=[]):
        data = flip_it(data, axes)
        prediction = \
            patch_wise_prediction(model=model,
                                  data=np.expand_dims(data, 0),
                                  overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"] + [config["patch_depth"]]).squeeze()
        return flip_it(prediction, axes)

    predictions = []
    for axes in powerset([0, 1, 2]):
        print(axes)
        predictions += [predict_it(data, axes)]

    return predictions


def predict_flips(data, model, overlap_factor, config):
    data_max = data.max()
    data_min = data.min()

    order = 2
    predictions = []
    for _ in range(10):
        val_range = data_max - data_min
        contrast_min_val = data_min + 0.2 * np.random.uniform(-1, 1) * val_range
        contrast_max_val = data_max + 0.2 * np.random.uniform(-1, 1) * val_range
        curr_data = contrast_augment(data, contrast_min_val, contrast_max_val)

        rotate_factor = np.random.uniform(0, 360)
        predictions += \
            [ndimage.rotate(patch_wise_prediction(model=model,
                                                  data=ndimage.rotate(curr_data, rotate_factor, order=order, reshape=False)[np.newaxis, ...],
                                                  overlap_factor=overlap_factor,
                                                  patch_shape=config["patch_shape"] + [config["patch_depth"]]).squeeze(),
                            -rotate_factor, order=order, reshape=False)]
    return predictions


def save_nifti(data, path):
    nifti = get_image(data)
    nib.save(nifti, path)


def secondary_prediction(mask, vol, config2, model2_path=None,
                         preprocess_method2=None, norm_params2=None,
                         overlap_factor=0.9, augment2=None, num_augment=10, std=False):
    model2 = load_old_model(get_last_model_path(model2_path), config=config2)
    pred = mask
    bbox_start, bbox_end = find_bounding_box(pred)
    check_bounding_box(pred, bbox_start, bbox_end)
    padding = [16, 16, 8]
    if padding is not None:
        bbox_start = np.maximum(bbox_start - padding, 0)
        bbox_end = np.minimum(bbox_end + padding, mask.shape)
    data = vol.astype(np.float)[
           bbox_start[0]:bbox_end[0],
           bbox_start[1]:bbox_end[1],
           bbox_start[2]:bbox_end[2]
           ]

    data = preproc_and_norm(data, preprocess_method2, norm_params2)

    prediction, prediction_std = get_prediction(data, model2, augment=augment2, num_augments=num_augment, std=std,
                                                overlap_factor=overlap_factor, config=config2)

    padding2 = list(zip(bbox_start, np.array(vol.shape) - bbox_end))
    print(padding2)
    print(prediction.shape)
    prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)
    if prediction_std is not None:
        prediction_std = np.pad(prediction_std, padding2, mode='constant', constant_values=0)

    return prediction, prediction_std


def preproc_and_norm(data, preprocess_method=None, norm_params=None, scale=None, preproc=None):
    if preprocess_method is not None:
        print('Applying preprocess by {}...'.format(preprocess_method))
        if preprocess_method == 'window_1_99':
            data = window_intensities_data(data)
        else:
            raise Exception('Unknown preprocess: {}'.format(preprocess_method))

    if scale is not None:
        data = ndimage.zoom(data, scale)
    if preproc is not None:
        preproc_func = getattr(fetal_net.preprocess, preproc)
        data = preproc_func(data)

    # data = normalize_data(data, mean=data.mean(), std=data.std())
    if norm_params is not None and any(norm_params.values()):
        data = normalize_data(data, mean=norm_params['mean'], std=norm_params['std'])
    return data


def get_prediction(data, model, augment, num_augments, std, overlap_factor, config):
    prediction_std = None
    if augment is not None:
        if augment == 'all':
            predictions = predict_augment(data, model=model, overlap_factor=overlap_factor, config=config, num_augments=num_augments)
        elif augment == 'flip':
            predictions = predict_flips(data, model=model, overlap_factor=overlap_factor, config=config)
        else:
            raise ("Unknown augmentation {}".format(augment))
        prediction = np.std(predictions, axis=0)
        if std:
            prediction_std = np.std(predictions, axis=0).squeeze()
    else:
        prediction = \
            patch_wise_prediction(model=model,
                                  data=np.expand_dims(data, 0),
                                  overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"] + [config["patch_depth"]])
    prediction = prediction.squeeze()

    return prediction, prediction_std


def main(input_path, output_path, overlap_factor,
         config, model_path, preprocess_method=None, norm_params=None, augment=None, num_augment=0,
         config2=None, model2_path=None, preprocess_method2=None, norm_params2=None, augment2=None, num_augment2=0,
         z_scale=None, xy_scale=None, std=False):
    print(model_path)
    model = load_old_model(get_last_model_path(model_path), config=config)
    print('Loading nifti from {}...'.format(input_path))
    nifti = read_img(input_path)
    print('Predicting mask...')
    data = nifti.get_fdata().astype(np.float).squeeze()
    print('original_shape: ' + str(data.shape))
    scan_name = Path(input_path).name.split('.')[0]

    if (z_scale is None):
        z_scale = 1.0
    if (xy_scale is None):
        xy_scale = 1.0
    if z_scale != 1.0 or xy_scale != 1.0:
        data = ndimage.zoom(data, [xy_scale, xy_scale, z_scale])

    data = preproc_and_norm(data, preprocess_method, norm_params,
                            scale=config.get('scale_data', None),
                            preproc=config.get('preproc', None))

    save_nifti(data, os.path.join(output_path, scan_name + '_data.nii.gz'))

    data = np.pad(data, 3, 'constant', constant_values=data.min())

    print('Shape: ' + str(data.shape))
    prediction, prediction_std = get_prediction(data, model, augment=augment, num_augments=num_augment, std=std,
                                                overlap_factor=overlap_factor, config=config)
    # unpad
    prediction = prediction[3:-3, 3:-3, 3:-3]
    if std:
        prediction = prediction_std[3:-3, 3:-3, 3:-3]

    # revert to original size
#    if config.get('scale_data', None) is not None:
#        prediction = ndimage.zoom(prediction.squeeze(), np.divide([1, 1, 1], config.get('scale_data', None)), order=0)[
#            ..., np.newaxis]
#        if std:
#            prediction_std = ndimage.zoom(prediction_std.squeeze(), np.divide([1, 1, 1], config.get('scale_data', None)), order=0)[
#                ..., np.newaxis]

    save_nifti(prediction, os.path.join(output_path, scan_name + '_pred.nii.gz'))
    save_nifti(prediction_std, os.path.join(output_path, scan_name + '_std.nii.gz'))

    if z_scale != 1.0 or xy_scale != 1.0:
        prediction = ndimage.zoom(prediction.squeeze(), [1.0 / xy_scale, 1.0 / xy_scale, 1.0 / z_scale], order=1)[..., np.newaxis]

    print('Post-processing mask...')
    if prediction.shape[-1] > 1:
        prediction = prediction[..., 1]
    prediction = prediction.squeeze()
    mask = process_pred(prediction, gaussian_std=0.5, threshold=0.5)  # .astype(np.uint8)

    if config2 is not None:
        nifti = read_img(input_path)
        prediction, prediction_std = secondary_prediction(mask, vol=nifti.get_fdata().astype(np.float),
                                                          config2=config2, model2_path=model2_path,
                                                          preprocess_method2=preprocess_method2, norm_params2=norm_params2,
                                                          overlap_factor=overlap_factor, augment2=augment2, num_augment=num_augment2, std=std)
        save_nifti(prediction, os.path.join(output_path, scan_name + 'pred_roi.nii.gz'))
        if prediction_std is not None:
            save_nifti(prediction_std, os.path.join(output_path, scan_name + 'std_roi.nii.gz'))

    print('Saving to {}'.format(output_path))
    print('Finished.')


def get_params(config_dir):
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        __config = json.load(f)
    with open(os.path.join(config_dir, 'norm_params.json'), 'r') as f:
        __norm_params = json.load(f)
    __model_path = os.path.join(config_dir, os.path.basename(__config['model_file']))
    return __config, __norm_params, __model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nii", help="specifies mat file dir path",
                        type=str, required=True)
    parser.add_argument("--output_folder", help="specifies mat file dir path",
                        type=str, required=True)
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.9)
    parser.add_argument("--z_scale", help="specifies overlap between prediction patches",
                        type=float, default=1)
    parser.add_argument("--xy_scale", help="specifies overlap between prediction patches",
                        type=float, default=1)
    parser.add_argument("--std", help="output std for prediction",
                        type=int, default=0)

    # Params for primary prediction
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--preprocess", help="what preprocess to do",
                        type=str, required=False, default=None)
    parser.add_argument("--augment", help="what augment to do",
                        type=str, required=False, default=None)  # one of 'flip, all'
    parser.add_argument("--num_augment", help="what augment to do",
                        type=int, required=False, default=0)  # one of 'flip, all'

    # Params for secondary prediction
    parser.add_argument("--config2_dir", help="specifies config dir path",
                        type=str, required=False, default=None)
    parser.add_argument("--preprocess2", help="what preprocess to do",
                        type=str, required=False, default=None)
    parser.add_argument("--augment2", help="what augment to do",
                        type=str, required=False, default=None)  # one of 'flip, all'
    parser.add_argument("--num_augment2", help="what augment to do",
                        type=int, required=False, default=0)  # one of 'flip, all'

    opts = parser.parse_args()

    Path(opts.output_folder).mkdir(exist_ok=True)

    # 1
    _config, _norm_params, _model_path = get_params(opts.config_dir)
    # 2
    if opts.config2_dir is not None:
        _config2, _norm_params2, _model2_path = get_params(opts.config2_dir)
    else:
        _config2, _norm_params2, _model2_path = None, None, None

    main(opts.input_nii, opts.output_folder, overlap_factor=opts.overlap_factor,
         config=_config, model_path=_model_path, preprocess_method=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
         num_augment=opts.num_augment,
         config2=_config2, model2_path=_model2_path, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2, augment2=opts.augment2,
         num_augment2=opts.num_augment2,
         z_scale=opts.z_scale, xy_scale=opts.xy_scale, std=opts.std)
