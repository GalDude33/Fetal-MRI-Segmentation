import numpy as np
import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, Activation, add, UpSampling2D, Conv2DTranspose, Flatten, AveragePooling2D, Conv3D, UpSampling3D, Conv3DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.models import Model
from keras_contrib.layers.normalization import InstanceNormalization, InputSpec

np.random.seed(seed=12345)


class CycleGAN():
    def __init__(self, image_shape=(256 * 1, 256 * 1, 1)):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = True #False

    # ===============================================================================
    # Architecture functions

    def ck(self, x, k, use_normalization):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def c7Ak3D(self, x, k):
        x = Conv3D(filters=k, kernel_size=(7, 7, 7), strides=1, padding='valid')(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk3D(self, x, k):
        x = Conv3D(filters=k, kernel_size=3, strides=(2, 2, 1), padding='same')(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def Rk3D(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding3D((1, 1, 1))(x0)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding3D((1, 1, 1))(x)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def uk3D(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling3D(size=(2, 2, 1))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding3D((1, 1, 1))(x)
            x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv3DTranspose(filters=k, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # ===============================================================================
    # Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        # x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        # out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('linear')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator3D(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding3D((3, 3, 3))(input_img)
        x = self.c7Ak3D(x, 32)
        # Layer 2
        x = self.dk3D(x, 64)
        # Layer 3
        x = self.dk3D(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk3D(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk3D(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk3D(x, 128)

        # Layer 13
        x = self.uk3D(x, 64)
        # Layer 14
        x = self.uk3D(x, 32)
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('linear')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0],
                s[1] + 2 * self.padding[0],
                s[2] + 2 * self.padding[1],
                s[3] + 2 * self.padding[2],
                s[4])

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')


def build_genAB(input_shape, name='genAB', **kwargs):
    return CycleGAN(image_shape=input_shape).modelGenerator(name=name)


def build_genAB_3D(input_shape, name='genAB', **kwargs):
    return CycleGAN(image_shape=input_shape).modelGenerator3D(name=name)
