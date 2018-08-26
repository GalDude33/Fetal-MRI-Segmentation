import glob
import os


def get_last_model_path(model_file_path):
    return sorted(glob.glob(model_file_path + '*.h5'), key=os.path.getmtime)[-1]
