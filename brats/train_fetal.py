import os
import glob

import fetal_net
from brats.utils import get_last_model_path
from fetal_net.data import write_data_to_file, open_data_file
from fetal_net.generator import get_training_and_validation_generators
from fetal_net.model.fetal_net import fetal_envelope_model
from fetal_net.training import load_old_model, train_model
from pathlib import Path

import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--overwrite_config", help="overwrite saved config",
                    action="store_true")
parser.add_argument("--config_dir", help="specifies config dir path",
                    type=str, required=True)
parser.add_argument("--split_dir", help="specifies config dir path",
                    type=str, required=True)
opts = parser.parse_args()

# Load previous config if exists
if Path(os.path.join(opts.config_dir, 'config.json')).exists() and not opts.overwrite_config:
    print('Loading previous config.json from {}'.format(opts.config_dir))
    with open(os.path.join(opts.config_dir, 'config.json')) as f:
        config = json.load(f)
else:
    config = dict()
    config["base_dir"] = opts.config_dir
    config["split_dir"] = opts.split_dir

    Path(config["base_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["split_dir"]).mkdir(parents=True, exist_ok=True)

    # config["image_shape"] = (256, 256, 108)  # This determines what shape the images will be cropped/resampled to.
    config["patch_shape"] = (128, 128)  # switch to None to train on the whole image
    config["patch_depth"] = 5
    config["truth_index"] = 2
    config["prev_truth_index"] = 1  # None for regular training
    config["prev_truth_size"] = 1  # None for regular training
    config["truth_downsample"] = None  # 96
    config["truth_crop"] = None  # if true will crop sample else resize

    config["model_name"] = 'isensee2017_model'  # 'unet_model_2d'

    config["labels"] = (1,)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    config["all_modalities"] = ["volume"]
    config["training_modalities"] = config[
        "all_modalities"]  # change this if you want to only use some of the modalities
    config["nb_channels"] = len(config["training_modalities"])
    config["input_shape"] = tuple(list(config["patch_shape"]) +
                                  [config["patch_depth"] + (
                                      config["prev_truth_size"] if config["prev_truth_index"] is not None else 0)])
    config["truth_channel"] = config["nb_channels"]

    config["batch_size"] = 1
    config["patches_per_img_per_batch"] = 1
    config["validation_batch_size"] = 2
    config["n_epochs"] = 300  # cutoff the training after this many epochs
    config[
        "patience"] = 3  # learning rate will be reduced after this many epochs if the validation loss is not improving
    config["early_stop"] = 25  # training will be stopped after this many epochs without the validation loss improving
    config["initial_learning_rate"] = 5e-4
    config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    config["validation_split"] = 0.90  # portion of the data that will be used for training

    config["categorical"] = False  # will make the target one_hot
    config["loss"] = 'dice'  # or 'dice_coeeficient'

    config["augment"] = {
        "flip": [0.5, 0.5, 0],  # augments the data by randomly flipping an axis during
        "permute": False,  # data shape must be a cube. Augments the data by permuting in various directions
        "translate": (10, 10, 5),  #
        "scale": 0.10,  # i.e 0.20 for 20%, std of scaling factor, switch to None if you want no distortion
        "rotate": (3, 3, 7),  # std of angle rotation, switch to None if you want no rotation
    }
    config["augment"] = config["augment"] if any(config["augment"].values()) else None
    config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
    config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
    config["skip_blank_train"] = False  # if True, then patches without any target will be skipped
    config["skip_blank_val"] = False  # if True, then patches without any target will be skipped
    config["drop_easy_patches_train"] = True
    config["drop_easy_patches_val"] = False

    config["data_file"] = os.path.join(config["base_dir"], "fetal_data.h5")
    config["model_file"] = os.path.join(config["base_dir"], "fetal_net_model")
    config["training_file"] = os.path.join(opts.split_dir, "training_ids.pkl")
    config["validation_file"] = os.path.join(opts.split_dir, "validation_ids.pkl")
    config["test_file"] = os.path.join(opts.split_dir, "test_ids.pkl")
    config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

    config['scans_dir'] = "../../Datasets/Cutted_to_fetus"

    with open(os.path.join(config["base_dir"], 'config.json'), mode='w') as f:
        json.dump(config, f, indent=2)


def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    # for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
    for subject_dir in sorted(glob.glob(os.path.join(config["scans_dir"], "*")),
                              key=os.path.basename):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        _, (mean, std) = write_data_to_file(training_files, config["data_file"], subject_ids=subject_ids)
        with open(os.path.join(config["base_dir"], 'norm_params.json'), mode='w') as f:
            json.dump({'mean': mean, 'std': std}, f)

    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and len(glob.glob(config["model_file"] + '*.h5')) > 0:
        model_path = get_last_model_path(config["model_file"])
        print('Loading model from: {}'.format(model_path))
        model = load_old_model(model_path)
    else:
        # instantiate new model
        model_func = getattr(fetal_net.model, config['model_name'])
        model = model_func(input_shape=config["input_shape"],
                           initial_learning_rate=config["initial_learning_rate"],
                           loss_function=config["loss"])
    model.summary()

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        test_keys_file=config["test_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=(*config["patch_shape"], config["patch_depth"]),
        validation_batch_size=config["validation_batch_size"],
        augment=config["augment"],
        skip_blank_train=config["skip_blank_train"],
        skip_blank_val=config["skip_blank_val"],
        truth_index=config["truth_index"],
        prev_truth_index=config["prev_truth_index"],
        prev_truth_size=config["prev_truth_size"],
        truth_downsample=config["truth_downsample"],
        truth_crop=config["truth_crop"],
        patches_per_img_per_batch=config["patches_per_img_per_batch"],
        categorical=config["categorical"],
        drop_easy_patches_train=config["drop_easy_patches_train"],
        drop_easy_patches_val=config["drop_easy_patches_val"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                output_folder=config["base_dir"])
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
