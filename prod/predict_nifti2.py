import argparse
import json
import os

import numpy as np
from nilearn.image import new_img_like

from brats.utils import get_last_model_path
from fetal_net.normalize import normalize_data
from fetal_net.postprocess import postprocess_prediction as process_pred
from fetal_net.prediction import patch_wise_prediction
from fetal_net.training import load_old_model
from brats.preprocess import window_intensities_data

from fetal_net.utils.cut_relevant_areas import find_bounding_box, check_bounding_box
from fetal_net.utils.utils import read_img, get_image
import nibabel as nib


def save_nifti(data, path):
    nifti = get_image(data)
    nib.save(nifti, path)


def secondary_prediction(mask, vol, config2, model2_path=None,
                         preprocess_method2=None, norm_params2=None,
                         overlap_factor=0.9):
    model2 = load_old_model(get_last_model_path(model2_path))
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

    prediction = \
        patch_wise_prediction(model=model2,
                              data=np.expand_dims(data, 0),
                              overlap_factor=overlap_factor,
                              patch_shape=config2["patch_shape"] + [config2["patch_depth"]])
    prediction = prediction.squeeze()
    padding2 = list(zip(bbox_start, np.array(vol.shape) - bbox_end))
    print(padding2)
    print(prediction.shape)
    prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)
    return prediction


def preproc_and_norm(data, preprocess_method, norm_params):
    if preprocess_method is not None:
        print('Applying preprocess by {}...'.format(preprocess_method))
        if preprocess_method == 'window_1_99':
            data = window_intensities_data(data)
        else:
            raise Exception('Unknown preprocess: {}'.format(preprocess_method))

    if norm_params is not None and any(norm_params.values()):
        data = normalize_data(data, mean=norm_params['mean'], std=norm_params['std'])
    return data


def main(input_path, output_path, overlap_factor,
         config, model_path, preprocess_method=None, norm_params=None,
         config2=None, model2_path=None, preprocess_method2=None, norm_params2=None):
    print(model_path)
    model = load_old_model(get_last_model_path(model_path))
    print('Loading mat from {}...'.format(input_path))
    nifti = read_img(input_path)
    print('Predicting mask...')
    data = nifti.get_fdata().astype(np.float)

    data = preproc_and_norm(data, preprocess_method, norm_params)

    prediction = \
        patch_wise_prediction(model=model,
                              data=np.expand_dims(data, 0),
                              overlap_factor=overlap_factor,
                              patch_shape=config["patch_shape"] + [config["patch_depth"]])
    save_nifti(prediction, os.path.join(output_path, 'pred.nii'))

    print('Post-processing mask...')
    if prediction.shape[-1] > 1:
        prediction = prediction[..., 1]
    prediction = prediction.squeeze()
    print("Storing prediction in [7-9], 7 should be the best...")
    mask = process_pred(prediction, gaussian_std=0.5, threshold=0.5)  # .astype(np.uint8)

    if config2 is not None:
        print('Making secondary prediction... [6]')
        prediction = secondary_prediction(mask, vol=nifti.get_fdata().astype(np.float),
                                          config2=config2, model2_path=model2_path,
                                          preprocess_method2=preprocess_method2, norm_params2=norm_params2,
                                          overlap_factor=0.9)
        save_nifti(prediction, os.path.join(output_path, 'pred_roi.nii'))

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

    # Params for primary prediction
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--preprocess", help="what preprocess to do",
                        type=str, required=False, default=None)

    # Params for secondary prediction
    parser.add_argument("--config2_dir", help="specifies config dir path",
                        type=str, required=False, default=None)
    parser.add_argument("--preprocess2", help="what preprocess to do",
                        type=str, required=False, default=None)

    opts = parser.parse_args()

    # 1
    _config, _norm_params, _model_path = get_params(opts.config_dir)
    # 2
    if opts.config2_dir is not None:
        _config2, _norm_params2, _model2_path = get_params(opts.config2_dir)
    else:
        _config2, _norm_params2, _model2_path = None, None, None

    main(opts.input_nii, opts.output_folder, overlap_factor=opts.overlap_factor,
         config=_config, model_path=_model_path, preprocess_method=opts.preprocess, norm_params=_norm_params,
         config2=_config2, model2_path=_model2_path, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2)
