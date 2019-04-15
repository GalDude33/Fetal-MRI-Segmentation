import glob
import json
import os

from collections import defaultdict

import fetal_net.preprocess
from fetal_net.data import write_data_to_file


def create_data_file(config):
    training_files, subject_ids = fetch_training_data_files(config, return_subject_ids=True)
    if config.get('preproc', None) is not None:
        preproc_func = getattr(fetal_net.preprocess, config['preproc'])
    else:
        preproc_func = None
    _, (mean, std) = write_data_to_file(training_files, config["data_file"], subject_ids=subject_ids,
                                        normalize=config['normalization'], scale=config.get('scale_data', None),
                                        preproc=preproc_func)
    with open(os.path.join(config["base_dir"], 'norm_params.json'), mode='w') as f:
        json.dump({'mean': mean, 'std': std}, f)


def fetch_training_data_files(config, return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()

    for subject_dir in sorted(glob.glob(os.path.join(config["scans_dir"], "*")),
                              key=os.path.basename):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"] + (
                config["weight_mask"] if config["weight_mask"] is not None else []):
            subject_files.append(os.path.join(subject_dir, modality + ".nii" + config["ext"]))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def get_last_model_path(model_file_path):
    return sorted(glob.glob(model_file_path + '*.h5'), key=os.path.getmtime)[-1]


class AttributeDict(defaultdict):
    def __init__(self, **kwargs):
        super(AttributeDict, self).__init__(AttributeDict, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

def set_gpu_mem_growth():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
