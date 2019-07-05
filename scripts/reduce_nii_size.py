import argparse
import nibabel as nib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", help="source", type=str, required=True)
opts = parser.parse_args()

src_dir = opts.src

ext = '.gz'

for p in Path(src_dir).glob('**/prediction.nii.gz'):
    orig_size = p.stat().st_size

    pred = nib.load(str(p)).get_data()

    pred[:] = (pred * 100).astype(int) / 100.0

    # entropy 1
    nib.save(pred, str(p))
    after_size = p.stat().st_size

    print(p.parent, orig_size, after_size)
