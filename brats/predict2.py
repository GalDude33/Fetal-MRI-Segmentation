import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib

from brats.utils import get_last_model_path
from fetal_net.normalize import normalize_data
from fetal_net.prediction import run_validation_cases, patch_wise_prediction
import argparse

from fetal_net.training import load_old_model
from fetal_net.utils.cut_relevant_areas import find_bounding_box, cut_bounding_box, check_bounding_box


def main(pred_dir, config, split='test', overlap_factor=1):
    padding = [16, 16, 8]
    prediction2_dir = os.path.abspath(os.path.join(config['base_dir'], 'predictions2', split))
    for sample_folder in glob(os.path.join(pred_dir, split, '*')):
        volume_path = os.path.join(sample_folder, 'volume.nii')
        mask_path = os.path.join(sample_folder, 'prediction.nii')

        subject_id = Path(sample_folder).name
        dest_folder = os.path.join(prediction2_dir, subject_id)
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        volume = nib.load(volume_path)
        orig_volume_shape = np.array(volume.get_data().shape)
        nib.save(volume, os.path.join(dest_folder, Path(volume_path).name))

        mask = nib.load(mask_path)
        bbox_start, bbox_end = find_bounding_box(mask.get_data())
        check_bounding_box(mask, bbox_start, bbox_end)
        if padding is not None:
            bbox_start = np.maximum(bbox_start - padding, 0)
            bbox_end = np.minimum(bbox_end + padding, mask.shape)
        volume = cut_bounding_box(volume, bbox_start, bbox_end).get_data()

        model = load_old_model(get_last_model_path(config["model_file"]))

        with open(os.path.join(opts.config_dir, 'norm_params.json'), 'r') as f:
            norm_params = json.load(f)
        if norm_params is not None and any(norm_params.values()):
            volume = normalize_data(volume, mean=norm_params['mean'], std=norm_params['std'])
        prediction = patch_wise_prediction(
            model=model, data=volume,
            patch_shape=config["patch_shape"] + [config["patch_depth"]],
            overlap_factor=overlap_factor
        )

        padding2 = list(zip(bbox_start, orig_volume_shape - bbox_end))
        prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)
        nib.save(prediction, os.path.join(dest_folder, Path(mask_path).name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--split", help="What split to predict on? (test/val)",
                        type=str, default='test')
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=1)
    opts = parser.parse_args()

    with open(os.path.join(opts.config_dir, 'config.json')) as f:
        config = json.load(f)

    main(config, opts.split, opts.overlap_factor)
