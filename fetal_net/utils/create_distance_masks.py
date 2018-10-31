import glob
import os
from pathlib import Path

import nibabel as nib
from scipy import ndimage

dataset_folder = ''
ext = '.gz'
sampling=(0.4, 0.4, 5.0)

for mask_path in glob.glob(os.path.join(dataset_folder, '*', 'truth.nii'+ext)):
    mask = nib.load(mask_path)

    dists = ndimage.morphology.distance_transform_edt(mask, sampling=sampling)
    dists_inv = ndimage.morphology.distance_transform_edt(1-mask, sampling=sampling)
    total_dists = dists + dists_inv

    nib.save(total_dists, os.path.join(Path(mask_path).parent, 'dists.nii.gz'))