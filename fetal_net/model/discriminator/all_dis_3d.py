import keras.backend as K
from keras import Model, Input
from keras.layers import BatchNormalization, Conv3D, Dense, GlobalAveragePooling3D, LeakyReLU
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras_contrib.layers import InstanceNormalization


def discriminator_image_3d(input_shape=(None, 2, 64, 128, 128),
                           n_base_filters=16,
                           optimizer=Adam, initial_learning_rate=5e-4,
                           depth=5, dropout_rate=0.3, **kargs):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    kernel_size = 3  # kernel size
    stride_size = 2  # stride
    padding = 'same'  # 'valid'
    scale_only_xy = 1

    inputs = Input(input_shape)

    conv = inputs
    for level in range(scale_only_xy):
        conv = conv_block(conv, level, 16, kernel_size, padding, (stride_size, stride_size, 1))
    for level in range(depth - scale_only_xy):
        conv = conv_block(conv, level, 16, kernel_size, padding, stride_size)

    gap = GlobalAveragePooling3D()(conv)
    outputs = Dense(1, activation='sigmoid')(gap)

    d = Model(inputs, outputs, name='Discriminator')

    def d_loss(y_true, y_pred):
        L = binary_crossentropy(K.batch_flatten(y_true),
                                K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=optimizer(lr=initial_learning_rate, beta_1=0.5),
              loss=d_loss,
              metrics=['accuracy'])

    return d


def mini_conv_block(input_layer, n_filters, kernel_size, padding, strides):
    conv = Conv3D(n_filters, kernel_size=kernel_size, padding=padding, strides=strides)(input_layer)
    #conv = BatchNormalization(scale=False, axis=1)(conv)
    conv = InstanceNormalization(axis=1)(conv)
    conv = LeakyReLU()(conv)
    return conv


def conv_block(input_layer, level, n_base_filters, kernel_size, padding, strides):
    n_filters = (2 ** level) * n_base_filters
    conv = mini_conv_block(input_layer, n_filters, kernel_size, padding, strides)
    conv = mini_conv_block(conv, n_filters, kernel_size, padding, strides)
    return conv