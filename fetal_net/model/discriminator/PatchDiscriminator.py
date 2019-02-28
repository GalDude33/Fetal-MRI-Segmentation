from keras import Input, Model
from keras.layers import LeakyReLU, Conv2D, Conv3D
from keras_contrib.layers import InstanceNormalization


def build_discriminator(input_shape, df=64, conv=None):
    if conv is None:
        raise AssertionError("conv should be on of {Conv2D, Conv3D}")

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = conv(filters, kernel_size=f_size, strides=(2,2,1), padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=input_shape)

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = conv(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

    return Model(img, validity)


def build_discriminator_2d(input_shape, df=64, **kwargs):
    return build_discriminator(input_shape, df=df, conv=Conv2D)


def build_discriminator_3d(input_shape, df=32, **kwargs):
    return build_discriminator(input_shape, df=df, conv=Conv3D)