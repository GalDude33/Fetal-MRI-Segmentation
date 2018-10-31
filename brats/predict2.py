import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib

from brats.preprocess import window_intensities_data
from fetal_net.utils.utils import get_image
from fetal_net.postprocess import postprocess_prediction as process_pred
from brats.utils import get_last_model_path
from fetal_net.normalize import normalize_data
from fetal_net.prediction import run_validation_cases, patch_wise_prediction
import argparse

from fetal_net.training import load_old_model
from fetal_net.utils.cut_relevant_areas import find_bounding_box, cut_bounding_box, check_bounding_box

original_data_folder = '../Datasets/Fetus'


def main(pred_dir, config, split='test', overlap_factor=1, preprocess_method=None):
    padding = [16, 16, 8]
    prediction2_dir = os.path.abspath(os.path.join(config['base_dir'], 'predictions2', split))
    model = load_old_model(get_last_model_path(config["model_file"]))
    with open(os.path.join(opts.config_dir, 'norm_params.json'), 'r') as f:
        norm_params = json.load(f)

    for sample_folder in glob(os.path.join(pred_dir, split, '*')):
        mask_path = os.path.join(sample_folder, 'prediction.nii.gz')
        truth_path = os.path.join(sample_folder, 'truth.nii.gz')

        subject_id = Path(sample_folder).name
        dest_folder = os.path.join(prediction2_dir, subject_id)
        Path(dest_folder).mkdir(parents=True, exist_ok=True)

        truth = nib.load(truth_path)
        nib.save(truth, os.path.join(dest_folder, Path(truth_path).name))

        mask = nib.load(mask_path)
        mask = process_pred(mask.get_data(), gaussian_std=0.5, threshold=0.5)
        bbox_start, bbox_end = find_bounding_box(mask)
        check_bounding_box(mask, bbox_start, bbox_end)
        if padding is not None:
            bbox_start = np.maximum(bbox_start - padding, 0)
            bbox_end = np.minimum(bbox_end + padding, mask.shape)
        print("BBox: {}-{}".format(bbox_start, bbox_end))

        volume = nib.load(os.path.join(original_data_folder, subject_id, 'volume.nii'))
        orig_volume_shape = np.array(volume.get_data().shape)
        volume = cut_bounding_box(volume, bbox_start, bbox_end).get_data().astype(np.float)

        if preprocess_method is not None:
            print('Applying preprocess by {}...'.format(preprocess_method))
            if preprocess_method == 'window_1_99':
                volume = window_intensities_data(volume)
            else:
                raise Exception('Unknown preprocess: {}'.format(preprocess_method))

        if norm_params is not None and any(norm_params.values()):
            volume = normalize_data(volume, mean=norm_params['mean'], std=norm_params['std'])

        prediction = patch_wise_prediction(
            model=model, data=np.expand_dims(volume, 0),
            patch_shape=config["patch_shape"] + [config["patch_depth"]],
            overlap_factor=overlap_factor
        )
        prediction = prediction.squeeze()

        padding2 = list(zip(bbox_start, orig_volume_shape - bbox_end))
        print(padding2)
        prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)
        assert all([s1 == s2 for s1, s2 in zip(prediction.shape, orig_volume_shape)])
        prediction = get_image(prediction)
        nib.save(prediction, os.path.join(dest_folder, Path(mask_path).name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--preprocess", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--split", help="What split to predict on? (test/val)",
                        type=str, default='test')
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=1)
    opts = parser.parse_args()

    with open(os.path.join(opts.config_dir, 'config.json')) as f:
        config = json.load(f)

    main(opts.pred_dir, config, opts.split, opts.overlap_factor, opts.preprocess)
