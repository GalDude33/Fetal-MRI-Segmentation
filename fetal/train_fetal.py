import json
import os
import glob

import fetal_net
import fetal_net.preprocess
import fetal_net.metrics
from fetal.config_utils import get_config
from fetal.utils import get_last_model_path, fetch_training_data_files, create_data_file
from fetal_net.data import write_data_to_file, open_data_file
from fetal_net.generator import get_training_and_validation_generators
from fetal_net.model.fetal_net import fetal_envelope_model
from fetal_net.training import load_old_model, train_model

config = get_config()


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        create_data_file(config)

    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and len(glob.glob(config["model_file"] + '*.h5')) > 0:
        model_path = get_last_model_path(config["model_file"])
        print('Loading model from: {}'.format(model_path))
        model = load_old_model(model_path)
    else:
        # instantiate new model
        loss_func = getattr(fetal_net.metrics, config['loss'])
        model_func = getattr(fetal_net.model, config['model_name'])
        model = model_func(input_shape=config["input_shape"],
                           initial_learning_rate=config["initial_learning_rate"],
                           **{'dropout_rate': config['dropout_rate'],
                              'loss_function': loss_func,
                              'mask_shape': None if config["weight_mask"] is None else config["input_shape"],
                              # TODO: change to output shape
                              'old_model_path': config['old_model']})
        if not overwrite and len(glob.glob(config["model_file"] + '*.h5')) > 0:
            model_path = get_last_model_path(config["model_file"])
            print('Loading model from: {}'.format(model_path))
            model.load_weights(model_path)
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
        truth_size=config["truth_size"],
        prev_truth_index=config["prev_truth_index"],
        prev_truth_size=config["prev_truth_size"],
        truth_downsample=config["truth_downsample"],
        truth_crop=config["truth_crop"],
        patches_per_epoch=config["patches_per_epoch"],
        categorical=config["categorical"], is3d=config["3D"],
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
