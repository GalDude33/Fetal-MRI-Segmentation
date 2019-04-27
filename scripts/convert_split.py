import os
from glob import glob
from pathlib import Path

from fetal_net.utils import pickle_load
from fetal_net.utils.utils import list_dump

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_split", help="source split dir path (pkl)", type=str, required=True)
parser.add_argument("--dst_split", help="destination split dir path (txt)", type=str, required=True)
opts = parser.parse_args()

src_dir = opts.src_split  # './split_dir_pkl/'
dst_dir = opts.dst_split  # './split_dir_txt/'

for f_name in glob(os.path.join(src_dir, '*.pkl')):
    l = pickle_load(f_name)
    list_dump(l, os.path.join(dst_dir, Path(f_name).stem + '.txt'))
