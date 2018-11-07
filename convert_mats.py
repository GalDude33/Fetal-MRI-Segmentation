from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat
import os

src_dir = '../Datasets/brain_ax_mat'
dst_dir = '../Datasets/brain_ax'


def extract_mask(masks):
    real_masks = [_ for _ in masks[0] if _ != []]
    assert len(real_masks) == 1
    return real_masks[0]


for src_scan_path in glob(os.path.join(src_dir, '*.mat')):
    src_scan = loadmat(src_scan_path)

    scan_dst_dir = os.path.join(dst_dir, Path(src_scan_path).stem)
    try:
        Path(scan_dst_dir).mkdir(exist_ok=True)
        nib.save(nib.Nifti1Pair(extract_mask(src_scan['masks']), np.eye(4)),
                 os.path.join(scan_dst_dir, 'truth.nii.gz'))
        nib.save(nib.Nifti1Pair(src_scan['volume'], np.eye(4)),
                 os.path.join(scan_dst_dir, 'volume.nii.gz'))
    except AssertionError as e:
        print(src_scan['UID'])
