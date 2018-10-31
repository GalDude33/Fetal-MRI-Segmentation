from functools import partial

from keras import Model, Input
from keras.layers import BatchNormalization, Conv2D, Softmax, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop
from tensorflow import Tensor

from fetal_net.metrics import vod_coefficient


def fetal_envelope_model(input_shape=(5, 64, 64),
                         optimizer=RMSprop,
                         initial_learning_rate=5e-4,
                         loss_function=binary_crossentropy):
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    kernel_size = (3, 3)
    padding = 'same'
    batch_norm = True

    Conv2D_ = partial(Conv2D, kernel_size=kernel_size, padding=padding, data_format='channels_last')

    def conv_block(input_layer, batch_norm=batch_norm):
        output = Conv2D_(16, activation='relu')(input_layer)
        output = MaxPooling2D(data_format='channels_last', padding='same')(output)
        if batch_norm:
            output = BatchNormalization()(output)
        return output

    def fc_block(input_layer: Tensor, output_channels, batch_norm=batch_norm,
                 activation='tanh'):
        output = Conv2D_(output_channels,
                         kernel_size=input_layer.shape[1:3].as_list(),  # input_layer.output_shape[:-1],
                         padding='valid',
                         activation=activation)(input_layer)
        if batch_norm:
            output = BatchNormalization()(output)
        return output

    input_layer = Input(input_shape)

    # net = addConvBlock(net, opts, '1', [3 3], 5, 16, 'pool', true, 'relu', true, 'bnorm', bnorm);
    conv_block_1 = conv_block(input_layer)

    # net = addConvBlock(net, opts, '2', [3 3], 16, 16, 'pool', true, 'relu', true, 'bnorm', bnorm);
    conv_block_2 = conv_block(conv_block_1)

    # net = addConvBlock(net, opts, '3', [3 3], 16, 16, 'pool', true, 'relu', true, 'bnorm', bnorm);
    conv_block_3 = conv_block(conv_block_2)

    # net = addConvBlock(net, opts, '4', 'fc', 16, 1000, 'pool', false, 'relu', false, 'bnorm', bnorm);
    fc_block_1 = fc_block(conv_block_3, 1000)

    # net = addConvBlock(net, opts, '5', 'fc', 1000, 2, 'pool', false, 'relu', false, 'bnorm', false); 
    fc_block_2 = fc_block(fc_block_1, 2, batch_norm=False)

    # net.layers{end + 1} = struct('type', 'softmax');
    output_layer = Softmax(name='softmax_last_layer')(fc_block_2)

    model = Model(inputs=input_layer, output=output_layer)
    model.compile(optimizer=optimizer(lr=initial_learning_rate),
                  loss=loss_function,
                  metrics=['binary_accuracy', vod_coefficient])  # 'binary_crossentropy')#loss_function)
    return model

# Sequential Model
# model = Sequential()
# model.add(Conv2D_(16, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(AveragePooling2D())
# model.add(Conv2D_(16, (3, 3), padding='same', activation='relu'))
# model.add(AveragePooling2D())
# model.add(BatchNormalization())
# model.add(Conv2D_(16, (3, 3), padding='same', activation='relu'))
# model.add(AveragePooling2D())
# model.add(BatchNormalization())
# model.add(Conv2D_(1000, (3, 3), padding='same', activation='tanh'))
# model.add(BatchNormalization())
# model.add(Conv2D_(2, (3, 3), padding='same', activation='tanh'))
