import json
import argparse
import os
from pathlib import Path


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_config", help="overwrite saved config",
                        action="store_true")
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--split_dir", help="specifies config dir path",
                        type=str, required=False)
    opts = parser.parse_args()
    # Load previous config if exists
    if Path(os.path.join(opts.config_dir, 'config.json')).exists() and not opts.overwrite_config:
        print('Loading previous config.json from {}'.format(opts.config_dir))
        with open(os.path.join(opts.config_dir, 'config.json')) as f:
            config = json.load(f)
    else:
        config = dict()
        config["base_dir"] = opts.config_dir
        config["split_dir"] = './debug_split'
        config['scans_dir'] = '../../Datasets/brain_new_cutted_window_1_99'
        config['fake_scans_dir'] = '../../Datasets/brain_new_cutted_window_1_99'

        Path(config["base_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["split_dir"]).mkdir(parents=True, exist_ok=True)

        # Training params
        config["batch_size"] = 1
        config["validation_batch_size"] = 1  # most of times should be equal to "batch_size"
        config["patches_per_epoch"] = 800  # patches_per_epoch / batch_size = steps per epoch

        config["n_epochs"] = 50  # cutoff the training after this many epochs
        config["patience"] = 3  # learning rate will be reduced after this many epochs if the validation loss is not improving
        config["early_stop"] = 7  # training will be stopped after this many epochs without the validation loss improving
        config["initial_learning_rate"] = 1e-4
        config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
        config["validation_split"] = 0.90  # portion of the data that will be used for training %

        config["3D"] = False  # Enable for 3D Models
        if config["3D"]:
            # Model params (3D)
            config["patch_shape"] = (16, 16)  # switch to None to train on the whole image
            config["patch_depth"] = 16
            config["truth_index"] = 0
            config["truth_size"] = 16
            model_name = 'isensee'  # or 'unet'
        else:
            # Model params (2D) - should increase "batch_size" and "patches_per_epoch"
            config["patch_shape"] = (64, 64)  # switch to None to train on the whole image
            config["patch_depth"] = 5
            config["truth_index"] = 2
            config["truth_size"] = 1
            model_name = 'unet'  # or 'isensee'

        # choose model
        config["model_name"] = {
            '3D': {
                'unet': 'unet_model_3d',
                'isensee': 'isensee2017_model_3d'
            },
            '2D': {
                'unet': 'unet_model_2d',
                'isensee': 'isensee2017_model'
            }
        }['3D' if config["3D"] else '2D'][model_name]
        config["model_name"] = 'dis_net'

        # choose loss
        config["loss"] = {
            0: 'binary_crossentropy_loss',
            1: 'dice_coefficient_loss',
            2: 'focal_loss',
            3: 'dice_and_xent',
            4: 'dice_and_xent_mask'
        }[1]

        config["augment"] = {
            "flip": [0.5, 0.5, 0.5],  # augments the data by randomly flipping an axis during
            "permute": False,
            # NOT SUPPORTED (data shape must be a cube. Augments the data by permuting in various directions)
            "translate": (15, 15, 7),  #
            "scale": (0.1, 0.1, 0),  # i.e 0.20 for 20%, std of scaling factor, switch to None if you want no distortion
            # "iso_scale": {
            #     "max": 1
            # },
            "rotate": (0, 0, 90),  # std of angle rotation, switch to None if you want no rotation
            "poisson_noise": 1,
            "gaussian_filter": {
                "prob": 0.0,
                "max_sigma": 1
            },
            "contrast": {
                'prob': 0,
                'min_factor': 0.2,
                'max_factor': 0.1
            },
            # "piecewise_affine": {
            #     'scale': 2
            # },
            "elastic_transform": {
                'alpha': 5,
                'sigma': 10
            },
            # "intensity_multiplication": 0.2,
            "coarse_dropout": {
                "rate": 0.2,
                "size_percent": [0.10, 0.30],
                "per_channel": True
            },
            "gaussian_noise": {
                "prob": 0.5,
                "sigma": 0.05
            },
            "speckle_noise": {
                "prob": 0.5,
                "sigma": 0.05
            }
        }

        # If the model outputs smaller result (x,y)-wise than the input
        config["truth_downsample"] = None  # factor to downsample the ground-truth
        config["truth_crop"] = False  # if true will crop sample else resize
        config["categorical"] = False  # will make the target one_hot

        # Relevant only for previous slice truth training
        config["prev_truth_index"] = None  # None for regular training
        config["prev_truth_size"] = None  # None for regular training

        config["labels"] = (1,)  # the label numbers on the input image - currently only 1 label supported

        config["skip_blank_train"] = False  # if True, then patches without any target will be skipped
        config["skip_blank_val"] = False  # if True, then patches without any target will be skipped
        config["drop_easy_patches_train"] = False  # will randomly prefer balanced patches (50% 1, 50% 0)
        config["drop_easy_patches_val"] = False  # will randomly prefer balanced patches (50% 1, 50% 0)

        # Data normalization
        config['normalization'] = {
            0: False,
            1: 'all',
            2: 'each'
        }[1]  # Normalize by all or each data mean and std

        # add ".gz" extension if needed
        config["ext"] = ".gz"

        # Not relevant at the moment...
        config["dropout_rate"] = 0

        # Weight masks (currently supported only with isensee3d model and dice_and_xent_weigthed loss)
        config["weight_mask"] = None  # ["dists"] # or []

        # Auto set - do not touch
        config["augment"] = config["augment"] if any(config["augment"].values()) else None
        config["n_labels"] = len(config["labels"])
        config["all_modalities"] = ["volume"]
        config["training_modalities"] = config[
            "all_modalities"]  # change this if you want to only use some of the modalities
        config["nb_channels"] = len(config["training_modalities"])
        config["input_shape"] = tuple(list(config["patch_shape"]) +
                                      [config["patch_depth"] + (
                                          config["prev_truth_size"] if config["prev_truth_index"] is not None else 0)])
        config["truth_channel"] = config["nb_channels"]
        # Auto set - do not touch
        config["data_file"] = os.path.join(config["base_dir"], "fetal_data.h5")
        config["model_file"] = os.path.join(config["base_dir"], "fetal_net_model")
        config["training_file"] = os.path.join(config["split_dir"], "training_ids.pkl")
        config["validation_file"] = os.path.join(config["split_dir"], "validation_ids.pkl")
        config["test_file"] = os.path.join(config["split_dir"], "test_ids.pkl")
        config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
        config["scale_data"] = (0.33, 0.33, 1)

        config["preproc"] = {
            0: "laplace",
            1: "laplace_norm",
            2: "grad",
            3: "grad_norm"
        }[1]
        config["preproc"] = None

        if config['3D']:
            config["input_shape"] = [1] + list(config["input_shape"])

        # relevant only to NormNet
        config["old_model"] = '/home/galdude33/Lab/workspace/fetal_envelope2/brats/debug_normnet/old_model.h5'

        with open(os.path.join(config["base_dir"], 'config.json'), mode='w') as f:
            json.dump(config, f, indent=2)

    return config
