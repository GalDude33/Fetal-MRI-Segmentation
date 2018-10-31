from keras import Input, Model
from keras.optimizers import Adam

from fetal_net.metrics import dice_coefficient_loss, dice_coefficient, vod_coefficient
from fetal_net.model.unet3d.isensee2017 import isensee2017_model_3d
from fetal_net.training import load_old_model


def norm_net_model(input_shape=(1, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                   n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                   loss_function=dice_coefficient_loss, old_model_path=None):
    inputs = Input(input_shape)
    norm_net = isensee2017_model_3d(input_shape, n_base_filters, depth, dropout_rate,
                                    n_segmentation_levels, n_labels, optimizer, initial_learning_rate,
                                    loss_function, activation_name=None)
    norm_inputs = norm_net(inputs)

    seg_net = load_old_model(old_model_path)
    seg_net.trainable = False
    segmentation = seg_net(norm_inputs)

    metrics = ['binary_accuracy', vod_coefficient]
    if loss_function != dice_coefficient_loss:
        metrics += [dice_coefficient]
    model = Model(inputs=[inputs], outputs=segmentation, name='NormNetModel')
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    return model
