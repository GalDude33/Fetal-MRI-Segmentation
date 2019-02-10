import glob
import os

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.engine.network import Network
from keras.layers import Concatenate, Activation
from keras.optimizers import Adam
from tqdm import tqdm

import fetal_net
import fetal_net.metrics
import fetal_net.preprocess
from fetal.config_utils import get_config
from fetal.utils import get_last_model_path, create_data_file
from fetal_net.data import open_data_file
from fetal_net.generator import get_training_and_validation_generators
from fetal_net.model.fetal_net import fetal_envelope_model
from fetal_net.training import load_old_model

config = get_config()
if not "dis_model_name" in config:
    config["dis_model_name"] = "discriminator_image"
if not "dis_loss" in config:
    config["dis_loss"] = "binary_crossentropy_loss"


class Scheduler:
    def __init__(self, n_itrs_per_epoch_d, n_itrs_per_epoch_g, init_lr, lr_decay, lr_patience):
        self.init_dsteps = n_itrs_per_epoch_d
        self.init_gsteps = n_itrs_per_epoch_g
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience

        self.dsteps = self.init_dsteps
        self.gsteps = self.init_gsteps
        self.lr = self.init_lr
        self.steps_stuck = 0
        self.best_loss = np.inf

    def get_dsteps(self):
        return self.dsteps

    def get_gsteps(self):
        return self.gsteps

    def get_lr(self):
        return self.lr

    def update_steps(self, n_round, loss):
        if loss < self.best_loss:
            self.steps_stuck = 0
            self.best_loss = loss
        else:
            self.steps_stuck += 1

        if self.steps_stuck > self.lr_patience:
            self.lr *= self.lr_decay
            print('Reducing LR to {}'.format(self.lr))

        # if key in self.schedules['step_decay']:
        # self.dsteps = max(int(self.init_dsteps * self.schedules['step_decay'][key]), 1)
        # self.gsteps = max(int(self.init_gsteps * self.schedules['step_decay'][key]), 1)


def build_dsc(out_labels, outs):
    s = ''
    for l, o in zip(out_labels, outs):
        s = s + '{}={:.3f}, '.format(l, o)
    return s[:-2]+'|'


def input2discriminator(real_patches, real_segs, fake_segs, d_out_shape):
    real = np.concatenate((real_patches, real_segs), axis=1)
    fake = np.concatenate((real_patches, fake_segs), axis=1)

    d_x_batch = np.concatenate((real, fake), axis=0)

    # real : 1, fake : 0
    d_y_batch = 0.9 * np.ones([d_x_batch.shape[0]] + list(d_out_shape)[1:])
    d_y_batch[real.shape[0]:, ...] = 0.1

    return d_x_batch, d_y_batch


def input2gan(real_patches, real_segs, d_out_shape):
    g_x_batch = real_patches
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch = [np.ones([real_patches.shape[0]] + list(d_out_shape)[1:])*0.9,
                 real_segs]
    return g_x_batch, g_y_batch


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        create_data_file(config)

    data_file_opened = open_data_file(config["data_file"])

    seg_loss_func = getattr(fetal_net.metrics, config['loss'])
    dis_loss_func = getattr(fetal_net.metrics, config['dis_loss'])
    if not overwrite \
            and len(glob.glob(config["model_file"] + 'dis_*.h5')) > 0 \
            and len(glob.glob(config["model_file"] + 'gen_*.h5')) > 0:
        dis_model_path = get_last_model_path(config["model_file"] + 'dis_')
        gen_model_path = get_last_model_path(config["model_file"] + 'gen_')
        print('Loading dis model from: {}'.format(dis_model_path))
        print('Loading gen model from: {}'.format(gen_model_path))
        dis_model = load_old_model(dis_model_path)
        gen_model = load_old_model(gen_model_path)
    else:
        # instantiate new model
        seg_model_func = getattr(fetal_net.model, config['model_name'])
        gen_model = seg_model_func(input_shape=config["input_shape"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   **{'dropout_rate': config['dropout_rate'],
                                      'loss_function': seg_loss_func,
                                      'mask_shape': None if config["weight_mask"] is None else config[
                                          "input_shape"],
                                      'old_model_path': config['old_model']})

        dis_model_func = getattr(fetal_net.model, config['dis_model_name'])
        dis_model = dis_model_func(
            input_shape=[config["input_shape"][0] + config["n_labels"]] + config["input_shape"][1:],
            initial_learning_rate=config["initial_learning_rate"],
            **{'dropout_rate': config['dropout_rate'],
               'loss_function': dis_loss_func})

        if not overwrite \
                and len(glob.glob(config["model_file"] + 'd_*.h5')) > 0 \
                and len(glob.glob(config["model_file"] + 'g_*.h5')) > 0:
            dis_model_path = get_last_model_path(config["model_file"] + 'dis_')
            gen_model_path = get_last_model_path(config["model_file"] + 'gen_')
            print('Loading dis model from: {}'.format(dis_model_path))
            print('Loading gen model from: {}'.format(gen_model_path))
            dis_model = load_old_model(dis_model_path)
            gen_model = load_old_model(gen_model_path)
            dis_model.load_weights(dis_model_path)
            dis_model.load_weights(gen_model_path)

    gen_model.summary()
    dis_model.summary()

    # Build "frozen discriminator"
    frozen_dis_model = Network(
        dis_model.inputs,
        dis_model.outputs,
        name='frozen_discriminator'
    )
    frozen_dis_model.trainable = False

    inputs2 = Input(shape=config["input_shape"])
    segs = Activation(None, name='seg')(gen_model(inputs2))
    valid = Activation(None, name='dis')(frozen_dis_model(Concatenate(axis=1)([segs, inputs2])))
    combined_model = Model(inputs=[inputs2], outputs=[valid, segs])
    combined_model.compile(loss=['mse', seg_loss_func],
                           loss_weights=[1, 100],
                           optimizer=Adam(config["initial_learning_rate"]))
    combined_model.summary()

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

    # # run training
    # train_model(model=model,
    #             model_file=config["model_file"],
    #             training_generator=train_generator,
    #             validation_generator=validation_generator,
    #             steps_per_epoch=n_train_steps,
    #             validation_steps=n_validation_steps,
    #             initial_learning_rate=config["initial_learning_rate"],
    #             learning_rate_drop=config["learning_rate_drop"],
    #             learning_rate_patience=config["patience"],
    #             early_stopping_patience=config["early_stop"],
    #             n_epochs=config["n_epochs"],
    #             output_folder=config["base_dir"])

    # start training
    dis_steps = 1
    gen_steps = 10
    scheduler = Scheduler(dis_steps, gen_steps,
                          init_lr=config["initial_learning_rate"],
                          lr_patience=config["patience"],
                          lr_decay=config["learning_rate_drop"])

    best_loss = np.inf

    for epoch in range(config["n_epochs"]):
        postfix={'g': None, 'd': None} #, 'val_g': None, 'val_d': None}
        with tqdm(range(n_train_steps // gen_steps), dynamic_ncols=True,
                  postfix={'gen': None, 'dis': None, 'val_gen': None, 'val_dis': None, None: None}) as pbar:
            for n_round in pbar:
                # train D
                outputs = np.zeros(dis_model.metrics_names.__len__())
                for i in range(scheduler.get_dsteps()):
                    real_patches, real_segs = next(train_generator)
                    d_x_batch, d_y_batch = input2discriminator(real_patches, real_segs,
                                                               gen_model.predict(real_patches,
                                                                                 batch_size=config["batch_size"]),
                                                               dis_model.output_shape)
                    outputs += dis_model.train_on_batch(d_x_batch, d_y_batch)
                outputs /= scheduler.get_dsteps()
                postfix['d'] = build_dsc(dis_model.metrics_names, outputs)
                pbar.set_postfix(**postfix)

                # train G (freeze discriminator)
                outputs = np.zeros(combined_model.metrics_names.__len__())
                for i in range(scheduler.get_gsteps()):
                    real_patches, real_segs = next(train_generator)
                    g_x_batch, g_y_batch = input2gan(real_patches, real_segs, dis_model.output_shape)
                    outputs += combined_model.train_on_batch(g_x_batch, g_y_batch)
                outputs /= scheduler.get_gsteps()

                postfix['g'] = build_dsc(combined_model.metrics_names, outputs)
                pbar.set_postfix(**postfix)

            # evaluate on validation set
            dis_metrics = np.zeros(dis_model.metrics_names.__len__(), dtype=float)
            gen_metrics = np.zeros(gen_model.metrics_names.__len__(), dtype=float)
            evaluation_rounds = 10
            for n_round in range(evaluation_rounds):  # rounds_for_evaluation:
                # D
                val_patches, val_segs = next(validation_generator)
                d_x_test, d_y_test = input2discriminator(val_patches, val_segs,
                                                         gen_model.predict(val_patches,
                                                                           batch_size=config["validation_batch_size"]),
                                                         dis_model.output_shape)
                dis_metrics += dis_model.evaluate(d_x_test, d_y_test, batch_size=config["validation_batch_size"],
                                                  verbose=0)

                # G
                #gen_x_test, gen_y_test = input2gan(val_patches, val_segs, dis_model.output_shape)
                gen_metrics += gen_model.evaluate(val_patches, val_segs,
                                                  batch_size=config["validation_batch_size"],
                                                  verbose=0)

            dis_metrics /= float(evaluation_rounds)
            gen_metrics /= float(evaluation_rounds)
            # save the model and weights with the best validation loss
            if gen_metrics[0] < best_loss:
                with open(os.path.join(config["base_dir"], "g_{}_{:.3f}.json".format(n_round, gen_metrics[0])),
                          'w') as f:
                    f.write(gen_model.to_json())
                gen_model.save_weights(
                    os.path.join(config["base_dir"], "g_{}_{:.3f}.h5".format(n_round, gen_metrics[0])))

            postfix['val_d'] = build_dsc(dis_model.metrics_names, dis_metrics)
            postfix['val_g'] = build_dsc(combined_model.metrics_names, gen_metrics)
            pbar.set_postfix(**postfix)
            print('val_d: '+postfix['val_d'], end=' | ')
            print('val_g: '+postfix['val_g'])
            pbar.refresh()

            # update step sizes, learning rates
            scheduler.update_steps(n_round, gen_metrics[0])
            K.set_value(dis_model.optimizer.lr, scheduler.get_lr())
            K.set_value(combined_model.optimizer.lr, scheduler.get_lr())

    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
