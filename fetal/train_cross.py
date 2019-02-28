import glob
import os

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.engine.network import Network
from keras.layers import Activation, Concatenate, Reshape
from keras.optimizers import Adam
from tqdm import tqdm

import fetal_net
import fetal_net.metrics
import fetal_net.preprocess
from fetal.config_utils import get_config
from fetal.utils import get_last_model_path, create_data_file, set_gpu_mem_growth, build_dsc
from fetal_net.data import open_data_file, write_data_to_file
from fetal_net.generator import get_training_and_validation_generators
from fetal_net.model.fetal_net import fetal_envelope_model

set_gpu_mem_growth()

config = get_config()
if not "dis_model_name" in config:
    config["dis_model_name"] = "discriminator_image_3d"
if not "dis_loss" in config:
    config["dis_loss"] = "binary_crossentropy_loss"
if not "gen_steps" in config:
    config["gen_steps"] = 1
if not "dis_steps" in config:
    config["dis_steps"] = 1
if not "gd_loss_ratio" in config:
    config["gd_loss_ratio"] = 10

K.set_image_data_format('channels_last')


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
            self.steps_stuck = 0
            print('Reducing LR to {}'.format(self.lr))

        # if key in self.schedules['step_decay']:
        # self.dsteps = max(int(self.init_dsteps * self.schedules['step_decay'][key]), 1)
        # self.gsteps = max(int(self.init_gsteps * self.schedules['step_decay'][key]), 1)


def input2discriminator(real_patches, fake_patches, d_out_shape):
    d_x_batch = np.concatenate((real_patches, fake_patches), axis=0)

    # real : 1, fake : 0
    d_y_batch = np.clip(np.random.uniform(0.9, 1.0, size=[d_x_batch.shape[0]] + list(d_out_shape)[1:]),
                        a_min=0, a_max=1)
    d_y_batch[real_patches.shape[0]:, ...] = 1 - d_y_batch[real_patches.shape[0]:, ...]

    return d_x_batch, d_y_batch


def input2gan(real_patches, real_segs, d_out_shape):
    g_x_batch = [real_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch = [
        real_segs,
        np.clip(np.random.uniform(0.9, 1.0, size=[real_patches.shape[0]] + list(d_out_shape)[1:]), a_min=0, a_max=1)
    ]
    return g_x_batch, g_y_batch


def main(overwrite=False):
    # convert input images into an hdf5 file
    data_file_path = config['base_dir'] + '/{}_fetal_data.h5'

    A_data_file_path = data_file_path.format('A')
    B_data_file_path = data_file_path.format('B')
    if overwrite or (not os.path.exists(A_data_file_path)) or (not os.path.exists(B_data_file_path)):
        configA = config.copy()
        # configA['scans_dir'] = ''
        configA['data_file'] = A_data_file_path
        create_data_file(configA, name='A')

        # ugly patch
        configB = config.copy()
        # configB['scans_dir'] = ''
        configB['data_file'] = B_data_file_path
        create_data_file(configB, name='B')

    A_data_file_opened = open_data_file(A_data_file_path)
    B_data_file_opened = open_data_file(B_data_file_path)

    seg_loss_func = getattr(fetal_net.metrics, config['loss'])
    dis_loss_func = getattr(fetal_net.metrics, config['dis_loss'])

    # instantiate genAB model
    genAB_model_func = getattr(fetal_net.model, config['gen_model_name'])
    genAB_model = genAB_model_func(input_shape=config["input_shape_gen"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   activation='linear',
                                   pool_size=(2, 2, 1),
                                   **{'dropout_rate': config['dropout_rate'],
                                      'loss_function': seg_loss_func
                                      })


    # instantiate seg model
    segB_model_func = getattr(fetal_net.model, config['seg_model_name'])
    segB_model = segB_model_func(input_shape=config["input_shape_seg"],
                                 initial_learning_rate=config["initial_learning_rate"],
                                 activation='sigmoid',
                                 **{'dropout_rate': config['dropout_rate'],
                                    'loss_function': seg_loss_func
                                    })

    # dis_model_func = getattr(fetal_net.model, config['dis_model_name'])
    # dis_model = dis_model_func(
    #     input_shape=config["input_shape_gen"],
    #     initial_learning_rate=config["initial_learning_rate"],
    #     scale_only_xy=3,
    #     **{'dropout_rate': config['dropout_rate'],
    #        'loss_function': dis_loss_func})
    from fetal_net.model.discriminator import PatchDiscriminator
    dis_model = PatchDiscriminator.build_discriminator_3d(config["input_shape_gen"])
    dis_model.compile(optimizer=Adam(lr=config["initial_learning_rate"]*0.01),
                      loss=dis_loss_func, metrics=['mae'])

    if not overwrite \
            and len(glob.glob(config["model_file"] + '/segB_*.h5')) > 0 \
            and len(glob.glob(config["model_file"] + '/genAB_*.h5')) > 0:
        genAB_model_path = get_last_model_path(config["model_file"] + '/genAB_')
        print('Loading genAB model from: {}'.format(genAB_model_path))
        genAB_model.load_weights(genAB_model_path)

        segB_model_path = get_last_model_path(config["model_file"] + '/segB_')
        print('Loading segB model from: {}'.format(segB_model_path))
        segB_model.load_weights(segB_model_path)

    genAB_model.summary()
    segB_model.summary()
    dis_model.summary()

    # Build "frozen discriminator"
    frozen_dis_model = Network(
        dis_model.inputs,
        dis_model.outputs,
        name='frozen_discriminator'
    )
    frozen_dis_model.trainable = False

    inputs_A = Input(shape=config["input_shape_gen"])
    B_fake = Activation(None, name='B_fake')(genAB_model(inputs_A))
    B_seg = Activation(None, name='B_seg')(segB_model(Reshape(config["input_shape_seg"])(B_fake)))
    B_valid = Activation(None, name='dis')(frozen_dis_model(B_fake))
    combined_model = Model(inputs=[inputs_A],
                           outputs=[B_seg, B_valid])
    combined_model.compile(loss=[seg_loss_func, 'binary_crossentropy'],
                           loss_weights=[config["gd_loss_ratio"], 1],
                           optimizer=Adam(config["initial_learning_rate"]))
    combined_model.summary()

    # get training and testing generators
    A_train_generator, A_validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        A_data_file_opened,
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

    # get training and testing generators
    B_train_generator, B_validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        B_data_file_opened,
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
        val_augment=config["augment"],
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

    # start training
    scheduler = Scheduler(config["dis_steps"], config["gen_steps"],
                          init_lr=config["initial_learning_rate"],
                          lr_patience=config["patience"],
                          lr_decay=config["learning_rate_drop"])

    best_loss = np.inf
    for epoch in range(config["n_epochs"]):
        postfix = {'g': None, 'd': None}  # , 'val_g': None, 'val_d': None}
        with tqdm(range(n_train_steps // config["gen_steps"]), dynamic_ncols=True,
                  postfix={'gen': None, 'dis': None, 'val_gen': None, 'val_dis': None, None: None}) as pbar:
            for n_round in pbar:
                # train D
                outputs = np.zeros(dis_model.metrics_names.__len__())
                for i in range(scheduler.get_dsteps()):
                    A_patches, _ = next(A_train_generator)
                    B_real_patches, _ = next(B_train_generator)
                    d_x_batch, d_y_batch = input2discriminator(B_real_patches,
                                                               genAB_model.predict(A_patches,
                                                                                   batch_size=config["batch_size"]),
                                                               dis_model.output_shape)
                    outputs += dis_model.train_on_batch(d_x_batch, d_y_batch)
                if scheduler.get_dsteps():
                    outputs /= scheduler.get_dsteps()
                    postfix['d'] = build_dsc(dis_model.metrics_names, outputs)
                    pbar.set_postfix(**postfix)

                # train G (freeze discriminator)
                outputs = np.zeros(combined_model.metrics_names.__len__())
                for i in range(scheduler.get_gsteps()):
                    real_patches, real_segs = next(A_train_generator)
                    g_x_batch, g_y_batch = input2gan(real_patches, real_segs, dis_model.output_shape)
                    outputs += combined_model.train_on_batch(g_x_batch, g_y_batch)
                outputs /= scheduler.get_gsteps()

                postfix['g'] = build_dsc(combined_model.metrics_names, outputs)
                pbar.set_postfix(**postfix)

            # evaluate on validation set
            dis_metrics = np.zeros(dis_model.metrics_names.__len__(), dtype=float)
            gen_metrics = np.zeros(genAB_model.metrics_names.__len__(), dtype=float)
            evaluation_rounds = n_validation_steps
            for n_round in range(evaluation_rounds):  # rounds_for_evaluation:
                A_val_patches, A_val_segs = next(A_validation_generator)
                B_val_patches, _ = next(B_train_generator)

                # D
                if scheduler.get_dsteps() > 0:
                    d_x_test, d_y_test = \
                        input2discriminator(B_val_patches,
                                            genAB_model.predict(A_val_patches,
                                                                batch_size=config[
                                                                    "validation_batch_size"]),
                                            dis_model.output_shape)
                    dis_metrics += dis_model.evaluate(d_x_test, d_y_test,
                                                      batch_size=config["validation_batch_size"],
                                                      verbose=0)

                # G
                gen_x_test, gen_y_test = input2gan(A_val_patches, A_val_segs, dis_model.output_shape)
                gen_metrics += \
                    combined_model.evaluate(gen_x_test, gen_y_test,
                                            batch_size=config["validation_batch_size"],
                                            verbose=0)

            dis_metrics /= float(evaluation_rounds)
            gen_metrics /= float(evaluation_rounds)
            # save the model and weights with the best validation loss
            if gen_metrics[0] < best_loss:
                best_loss = gen_metrics[0]
                print('Saving Model...')
                with open(os.path.join(config["base_dir"], "genAB_{}_{:.3f}.json".format(epoch, gen_metrics[0])), 'w') as f:
                    f.write(genAB_model.to_json())
                genAB_model.save_weights(os.path.join(config["base_dir"], "genAB_{}_{:.3f}.h5".format(epoch, gen_metrics[0])))

                with open(os.path.join(config["base_dir"], "segB_{}_{:.3f}.json".format(epoch, gen_metrics[0])), 'w') as f:
                    f.write(segB_model.to_json())
                segB_model.save_weights(os.path.join(config["base_dir"], "segB_{}_{:.3f}.h5".format(epoch, gen_metrics[0])))

            postfix['val_d'] = build_dsc(dis_model.metrics_names, dis_metrics)
            postfix['val_g'] = build_dsc(genAB_model.metrics_names, gen_metrics)
            # pbar.set_postfix(**postfix)
            print('val_d: ' + postfix['val_d'], end=' | ')
            print('val_g: ' + postfix['val_g'])
            # pbar.refresh()

            # update step sizes, learning rates
            scheduler.update_steps(epoch, gen_metrics[0])
            K.set_value(dis_model.optimizer.lr, scheduler.get_lr())
            K.set_value(combined_model.optimizer.lr, scheduler.get_lr())

    A_data_file_opened.close()
    B_data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
