import glob
import os

import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.engine.network import Network
from keras.layers import Activation, Concatenate, Lambda
from keras.optimizers import Adam
from tqdm import tqdm

import fetal_net
import fetal_net.metrics
import fetal_net.preprocess
from fetal.config_utils import get_config
from fetal.utils import get_last_model_path, create_data_file, set_gpu_mem_growth, Scheduler, build_dsc
from fetal_net.data import open_data_file
from fetal_net.generator import get_training_and_validation_generators
from fetal_net.model.discriminator import PatchDiscriminator
from fetal_net.model.fetal_net import fetal_envelope_model

set_gpu_mem_growth()

config = get_config()
if not "dis_model_name" in config:
    config["dis_model_name"] = "discriminator_image"
if not "dis_loss" in config:
    config["dis_loss"] = "binary_crossentropy_loss"
if not "gen_steps" in config:
    config["gen_steps"] = 1
if not "dis_steps" in config:
    config["dis_steps"] = 1
if not "gd_loss_ratio" in config:
    config["gd_loss_ratio"] = 10
if not "merge_method" in config:
    config["merge_method"] = 'mul_merge'  #'mul_merge'


def add_noise_to_segs(segs):
    if np.random.choice([True, False]):
        segs = segs.astype(np.float32)
        segs += np.random.normal(0, 0.025, segs.shape)
        segs *= np.random.normal(1, 0.025, segs.shape)
        segs = np.clip(segs, a_min=0, a_max=1)
    return segs


def mul_merge_maps(r, s):
    r_plus = r * s
    r_minus = r * (1 - s)
    return np.concatenate((r_plus, r_minus), axis=-1)


def input2discriminator(real_patches, real_segs,
                        semi_patches, semi_segs,
                        d_out_shape, mul_merge=config['merge_method'], add_noise=False):
    if not config["3D"]:
        t_i = config["truth_index"]
        real_patches = real_patches[..., t_i:t_i+1]
        semi_patches = semi_patches[..., t_i:t_i+1]

    if add_noise:
        real_segs = add_noise_to_segs(real_segs)

    if mul_merge:
        real = mul_merge_maps(real_patches, real_segs)
        fake = mul_merge_maps(semi_patches, semi_segs)
    else:
        real = np.concatenate((real_patches, real_segs), axis=-1)
        fake = np.concatenate((semi_patches, semi_segs), axis=-1)

    d_x_batch = np.concatenate((real, fake), axis=0)

    # real : 1, fake : 0
    d_y_batch = np.clip(np.random.uniform(0.85, 0.95, size=[d_x_batch.shape[0]] + list(d_out_shape)[1:]),
                        a_min=0, a_max=1)
    d_y_batch[real.shape[0]:, ...] = 1 - d_y_batch[real.shape[0]:, ...]

    return d_x_batch, d_y_batch


def input2gan(real_patches, real_segs, semi_patches, d_out_shape):
    g_x_batch = [real_patches, semi_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch = [
        real_segs,
        # Real
        np.clip(np.random.uniform(0.85, 0.95, size=[real_patches.shape[0]] + list(d_out_shape)[1:]), a_min=0, a_max=1),
        # Fake
        np.clip(np.random.uniform(0.85, 0.95, size=[real_patches.shape[0]] + list(d_out_shape)[1:]), a_min=0, a_max=1)
    ]
    return g_x_batch, g_y_batch


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        create_data_file(config)

    data_file_opened = open_data_file(config["data_file"])

    seg_loss_func = getattr(fetal_net.metrics, config['loss'])
    dis_loss_func = getattr(fetal_net.metrics, config['dis_loss'])

    # instantiate new model
    seg_model_func = getattr(fetal_net.model, config['seg_model_name'])
    gen_model = seg_model_func(input_shape=config["input_shape"],
                               initial_learning_rate=config["initial_learning_rate"],
                               **{'dropout_rate': config['dropout_rate'],
                                  'loss_function': seg_loss_func,
                                  'mask_shape': None if config["weight_mask"] is None else config[
                                      "input_shape"],
                                  'old_model_path': config['old_model']})

    if not overwrite \
            and len(glob.glob(config["model_file"] + 'g_*.h5')) > 0:
        gen_model_path = get_last_model_path(config["model_file"] + 'g_')
        print('Loading gen model from: {}'.format(gen_model_path))
        gen_model.load_weights(gen_model_path)

    # dis_model_func = getattr(fetal_net.model, config['dis_model_name'])
    # dis_model = dis_model_func(
    #     input_shape=[config["input_shape"][0] + config["n_labels"]] + config["input_shape"][1:],
    #     initial_learning_rate=config["initial_learning_rate"],
    #     **{'dropout_rate': config['dropout_rate'],
    #        'loss_function': dis_loss_func})
    dis_model = PatchDiscriminator.build_discriminator_2d(gen_model.output_shape[1:-1]+(2,))
    dis_model.compile(
        loss=['mse'],
        metrics=['mae'],
        optimizer=Adam(config["initial_learning_rate"]))

    gen_model.summary()
    dis_model.summary()

    # Build "frozen discriminator"
    frozen_dis_model = Network(
        dis_model.inputs,
        dis_model.outputs,
        name='frozen_discriminator'
    )
    frozen_dis_model.trainable = False

    inputs_real = Input(shape=config["input_shape"])
    inputs_fake = Input(shape=config["input_shape"])
    segs_real = Activation(None, name='s_R')(gen_model(inputs_real))
    segs_fake = Activation(None, name='s_S')(gen_model(inputs_fake))

    def mul_merge_k(seg, inp):
        return Lambda(lambda s_i: K.concatenate([s_i[0] * s_i[1],
                                                 (1-s_i[0]) * s_i[1]]))([seg, inp])

    def slice_channel_k(t, i):
        return Lambda(lambda t: t[..., i:i+1])(t)

    if config["merge_method"] == 'mul_merge':
        real_dis_input = mul_merge_k(segs_real, slice_channel_k(inputs_real, config["truth_index"]))
        semi_dis_input = mul_merge_k(segs_fake, slice_channel_k(inputs_fake, config["truth_index"]))
    else:
        real_dis_input = Concatenate()([segs_real, slice_channel_k(inputs_real, config["truth_index"])])
        semi_dis_input = Concatenate()([segs_fake, slice_channel_k(inputs_fake, config["truth_index"])])

    real_valid = Activation(None, name='dR')(frozen_dis_model(real_dis_input))
    semi_valid = Activation(None, name='dS')(frozen_dis_model(semi_dis_input))

    combined_model = Model(inputs=[inputs_real, inputs_fake],
                           outputs=[segs_real, real_valid, semi_valid])
    combined_model.compile(loss=[seg_loss_func, 'mse', 'mse'],
                           loss_weights=[1, config["gd_loss_ratio"], config["gd_loss_ratio"]],
                           metrics=['mae'],
                           optimizer=Adam(config["initial_learning_rate"]))
    combined_model.summary()

    gen_params = dict(batch_size=config["batch_size"],
                      data_split=config["validation_split"],
                      overwrite=overwrite,
                      validation_keys_file=config["validation_file"],
                      training_keys_file=config["training_file"],
                      test_keys_file=config["test_file"],
                      n_labels=config["n_labels"],
                      labels=config["labels"],
                      patch_shape=(*config["patch_shape"], config["patch_depth"]),
                      validation_batch_size=config["validation_batch_size"],
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
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        augment=config["augment"],
        **gen_params
    )

    # get training and testing generators
    _, semi_generator, _, _ = get_training_and_validation_generators(
        data_file_opened,
        val_augment=config["augment"],
        **gen_params)

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
                    real_patches, _ = next(train_generator)
                    semi_patches, _ = next(semi_generator)
                    d_x_batch, d_y_batch = input2discriminator(real_patches,
                                                               gen_model.predict(real_patches,
                                                                                 batch_size=config["batch_size"]),
                                                               # Fake:
                                                               semi_patches,
                                                               gen_model.predict(semi_patches,
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
                    real_patches, real_segs = next(train_generator)
                    semi_patches, _ = next(validation_generator)
                    g_x_batch, g_y_batch = input2gan(real_patches, real_segs, semi_patches, dis_model.output_shape)
                    outputs += combined_model.train_on_batch(g_x_batch, g_y_batch)
                outputs /= scheduler.get_gsteps()

                postfix['g'] = build_dsc(combined_model.metrics_names, outputs)
                pbar.set_postfix(**postfix)

            # evaluate on validation set
            dis_metrics = np.zeros(dis_model.metrics_names.__len__(), dtype=float)
            gen_metrics = np.zeros(gen_model.metrics_names.__len__(), dtype=float)
            evaluation_rounds = n_validation_steps
            for n_round in range(evaluation_rounds):  # rounds_for_evaluation:
                val_patches, val_segs = next(validation_generator)
                semi_val_patches, _ = next(semi_generator)

                # D
                if scheduler.get_dsteps() > 0:
                    d_x_test, d_y_test = \
                        input2discriminator(val_patches,
                                            gen_model.predict(val_patches,
                                                              batch_size=config["validation_batch_size"]),
                                            # Semi
                                            semi_val_patches,
                                            gen_model.predict(semi_val_patches,
                                                              batch_size=config["validation_batch_size"]),
                                            dis_model.output_shape)
                    dis_metrics += dis_model.evaluate(d_x_test, d_y_test, batch_size=config["validation_batch_size"],
                                                      verbose=0)

                # G
                # gen_x_test, gen_y_test = input2gan(val_patches, val_segs, dis_model.output_shape)
                gen_metrics += gen_model.evaluate(val_patches, val_segs,
                                                  batch_size=config["validation_batch_size"],
                                                  verbose=0)

            dis_metrics /= float(evaluation_rounds)
            gen_metrics /= float(evaluation_rounds)
            # save the model and weights with the best validation loss
            if gen_metrics[0] < best_loss:
                best_loss = gen_metrics[0]
                print('Saving Model...')
                with open(os.path.join(config["base_dir"], "g_{}_{:.3f}.json".format(epoch, gen_metrics[0])),
                          'w') as f:
                    f.write(gen_model.to_json())
                gen_model.save_weights(
                    os.path.join(config["base_dir"], "g_{}_{:.3f}.h5".format(epoch, gen_metrics[0])))

            postfix['val_d'] = build_dsc(dis_model.metrics_names, dis_metrics)
            postfix['val_g'] = build_dsc(gen_model.metrics_names, gen_metrics)

            print('val_d: ' + postfix['val_d'], end=' | ')
            print('val_g: ' + postfix['val_g'])

            # update step sizes, learning rates
            scheduler.update_steps(epoch, gen_metrics[0])
            K.set_value(dis_model.optimizer.lr, scheduler.get_lr())
            K.set_value(combined_model.optimizer.lr, scheduler.get_lr())

    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
