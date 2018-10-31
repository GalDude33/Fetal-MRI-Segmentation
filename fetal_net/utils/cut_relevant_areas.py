import os
from glob import glob
import nibabel as nib
import numpy as np
from pathlib import Path
from nilearn.image.image import _crop_img_to as crop_img_to, new_img_like


def main(src_dir, dst_dir, padding):
    for sample_folder in glob(os.path.join(src_dir, '*')):
        volume_path = os.path.join(sample_folder, 'volume.nii')
        mask_path = os.path.join(sample_folder, 'truth.nii')

        volume = nib.load(volume_path)
        volume = new_img_like(volume, volume.get_data(), affine=np.eye(4))
        mask = nib.load(mask_path)
        mask = new_img_like(volume, mask.get_data())

        bbox_start, bbox_end = find_bounding_box(mask.get_data())
        if padding is not None:
            bbox_start = np.maximum(bbox_start - padding, 0)
            bbox_end = np.minimum(bbox_end + padding, mask.shape)

        volume = cut_bounding_box(volume, bbox_start, bbox_end)
        mask = cut_bounding_box(mask, bbox_start, bbox_end)

        subject_id = Path(sample_folder).name
        dest_folder = os.path.join(dst_dir, subject_id)
        Path(dest_folder).mkdir(parents=True, exist_ok=True)

        nib.save(volume, os.path.join(dest_folder, Path(volume_path).name+'.gz'))
        nib.save(mask, os.path.join(dest_folder, Path(mask_path).name+'.gz'))


def cut_bounding_box(img, start, end):
    slices = [slice(s, e) for s, e in zip(start, end)]
    return crop_img_to(img, slices, copy=True)


def find_bounding_box(mask):
    coords = np.array(np.where(mask > 0))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    assert np.sum(mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]) == np.sum(mask)
    return start, end


def check_bounding_box(mask, start, end):
    return np.sum(mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]) == np.sum(mask)


if __name__ == '__main__':
    src_dir = '/home/galdude33/Lab/workspace/Datasets/brain_new'
    dst_dir = '/home/galdude33/Lab/workspace/Datasets/brain_new_cutted'

    padding = np.array([16, 16, 8])
    main(src_dir, dst_dir, padding)
