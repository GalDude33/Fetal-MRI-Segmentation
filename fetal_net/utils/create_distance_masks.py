import glob
import os
from pathlib import Path

import nibabel as nib
from scipy import ndimage
import numpy as np

dataset_folder = ''
ext = '.gz'
sampling=(0.4, 0.4, 3.0)

for mask_path in glob.glob(os.path.join(dataset_folder, '*', 'truth.nii'+ext)):
    print(Path(mask_path).parent.stem)

    mask = nib.load(mask_path).get_data()

    dists = ndimage.morphology.distance_transform_edt(mask, sampling=sampling)
    dists_inv = ndimage.morphology.distance_transform_edt(1-mask, sampling=sampling)
    total_dists = dists + dists_inv

    nib.save(nib.Nifti1Pair(total_dists, np.eye(4)), os.path.join(Path(mask_path).parent, 'dists.nii.gz'))