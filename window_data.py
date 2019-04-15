from fetal.preprocess import window_intensities

from glob import glob
from os import path
from shutil import copyfile

from pathlib import Path

src_dir = '../Datasets/brain_ax'
dst_dir = '../Datasets/brain_ax_window_1_99'
Path(dst_dir).mkdir(exist_ok=True, parents=False)
ext = '.gz'

for p in glob(path.join(src_dir, '*')):
    subject_dir = Path(p).name
    Path(path.join(dst_dir, subject_dir)).mkdir(exist_ok=True)
    window_intensities(in_file=path.join(src_dir, subject_dir, 'volume.nii' + ext),
                       out_file=path.join(dst_dir, subject_dir, 'volume.nii' + ext),
                       max_percent=99)

    copyfile(src=path.join(src_dir, subject_dir, 'truth.nii' + ext),
             dst=path.join(dst_dir, subject_dir, 'truth.nii' + ext))
    print(subject_dir)
