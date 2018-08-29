import argparse
import json
import os

import numpy as np

from brats.utils import get_last_model_path
from fetal_net.normalize import normalize_data
from fetal_net.postprocess import postprocess_prediction as process_pred
from fetal_net.prediction import patch_wise_prediction
from fetal_net.training import load_old_model
from brats.preprocess import window_intensities_data

from scipy.io import loadmat, savemat


def main(input_mat_path, output_mat_path, config, overlap_factor, model_path, preprocess_method=None, norm_params=None):
    print(model_path)
    model = load_old_model(get_last_model_path(model_path))
    print('Loading mat from {}...'.format(input_mat_path))
    mat = loadmat(input_mat_path)
    print('Predicting mask...')
    data = mat['volume'].astype(np.float)

    if preprocess_method is not None:
        print('Applying preprocess by {}...'.format(preprocess_method))
        if preprocess_method == 'window_1_99':
            data = window_intensities_data(data)
        else:
            raise Exception('Unknown preprocess: {}'.format(preprocess_method))

    if norm_params is not None and any(norm_params.values()):
        data = normalize_data(data, mean=norm_params['mean'], std=norm_params['std'])

    prediction = \
        patch_wise_prediction(model=model,
                              data=np.expand_dims(data, 0),
                              overlap_factor=overlap_factor,
                              patch_shape=config["patch_shape"] + [config["patch_depth"]])

    print('Post-processing mask...')
    if prediction.shape[-1] > 1:
        prediction = prediction[..., 1]
    prediction = prediction.squeeze()
    mat['masks'][0, 9] = \
        process_pred(prediction, gaussian_std=0, threshold=0.2)  # .astype(np.uint8)
    mat['masks'][0, 8] = \
        process_pred(prediction, gaussian_std=1, threshold=0.5)  # .astype(np.uint8)
    mat['masks'][0, 7] = \
        process_pred(prediction, gaussian_std=0.5, threshold=0.5)  # .astype(np.uint8)
    print('Saving mat to {}'.format(output_mat_path))
    savemat(output_mat_path, mat)
    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--input_mat", help="specifies mat file dir path",
                        type=str, required=True)
    parser.add_argument("--output_mat", help="specifies mat file dir path",
                        type=str, required=True)
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.9)
    parser.add_argument("--preprocess", help="what preprocess to do",
                        type=str, default=None)
    opts = parser.parse_args()

    with open(os.path.join(opts.config_dir, 'config.json'), 'r') as f:
        _config = json.load(f)
    with open(os.path.join(opts.config_dir, 'norm_params.json'), 'r') as f:
        _norm_params = json.load(f)

    _model_path = os.path.join(opts.config_dir, os.path.basename(_config['model_file']))
    main(opts.input_mat, opts.output_mat, _config, model_path=_model_path,
         preprocess_method=opts.preprocess, norm_params=_norm_params, overlap_factor=opts.overlap_factor)
