from keras import Input, Model
from keras.layers import LeakyReLU, Conv2D, Conv3D, Activation
from keras_contrib.layers import InstanceNormalization


def d_layer(layer_input, filters, conv, strides=2, f_size=4, normalization=True):
    """Discriminator layer"""
    d = conv(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if normalization:
        d = InstanceNormalization(axis=-1)(d)
    return d


def get_pool_size(i, only_xy_pool):
    if only_xy_pool == 0:
        return 2
    return (2, 2, 1) if i < only_xy_pool else (2, 2, 2)

def build_discriminator(input_shape, df=64, conv=None, depth=4, only_xy_pool=0):
    if conv is None:
        raise AssertionError("conv should be one of {Conv2D, Conv3D}")

    img = Input(shape=input_shape)

    d1 = img
    for i in range(0, depth):
        pool_size = get_pool_size(i, only_xy_pool)
        d1 = d_layer(d1, df * (2 ** i), conv, strides=pool_size, normalization=(i>0))

    validity = conv(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d1)
    return Model(img, validity)


def build_discriminator_2d(input_shape, df=64, depth=4, **kwargs):
    return build_discriminator(input_shape, df=df, depth=depth, conv=Conv2D)


def build_discriminator_3d(input_shape, df=32, depth=4, only_xy_layers=2, **kwargs):
    return build_discriminator(input_shape, df=df, depth=depth, only_xy_pool=only_xy_layers, conv=Conv3D)