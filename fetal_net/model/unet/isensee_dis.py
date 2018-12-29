from functools import partial

from keras.engine import Model
from keras.layers import Input, Add, UpSampling2D, Activation, SpatialDropout2D, Conv2D, Permute, LeakyReLU, \
    MaxPooling2D, Reshape
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate
from ...metrics import dice_coefficient_loss, dice_coefficient, \
    vod_coefficient

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=dice_coefficient_loss, activation_name="sigmoid", summation=False, **kargs):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
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
    metrics = ['binary_accuracy', vod_coefficient]
    if loss_function != dice_coefficient_loss:
        metrics += [dice_coefficient]

    inputs = Input(input_shape, name='main_input')

    def SegmentationModel():
        seg_input= Input(input_shape, name='seg_input')
        inputs_p = Permute((3, 1, 2), name='main_input_p')(seg_input)
        current_layer = inputs_p
        level_output_layers = list()
        level_filters = list()
        for level_number in range(depth):
            n_level_filters = (2 ** level_number) * n_base_filters
            level_filters.append(n_level_filters)

            if current_layer is inputs_p:
                in_conv = create_convolution_block(current_layer, n_level_filters)
            else:
                in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2))

            context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

            summation_layer = Add(name='sum_{}'.format(level_number))([in_conv, context_output_layer])

            level_output_layers.append(summation_layer)
            current_layer = summation_layer

        segmentation_layers = list()
        for level_number in range(depth - 2, -1, -1):
            up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
            concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
            localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
            current_layer = localization_output
            if level_number < n_segmentation_levels:
                segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

        if summation:
            output_layer = None
            for level_number in reversed(range(n_segmentation_levels)):
                segmentation_layer = segmentation_layers[level_number]
                if output_layer is None:
                    output_layer = segmentation_layer
                else:
                    output_layer = Add()([output_layer, segmentation_layer])

                if level_number > 0:
                    output_layer = UpSampling2D(size=(2, 2))(output_layer)
        else:
            output_layer = segmentation_layers[0]

        activation_block = Activation(activation_name)(output_layer)
        activation_block = Permute((2, 3, 1), name='SegLastLayer')(activation_block)

        seg_model = Model(inputs=seg_input, outputs=activation_block, name='SegmentationModel')
        return seg_model
    seg_model = SegmentationModel()

    encoder_model = Model(inputs=seg_model.input,
                          outputs=seg_model.get_layer('sum_{}'.format(depth - 1)).output,
                          name='EncoderModel')

    seg_res = Activation(name='seg_res', activation=None)(seg_model(inputs))

    def DiscriminatorModel(in_shape=(256, 4, 4)):
        in1 = Input(in_shape)
        conv1 = Conv2D(512, (3, 3), padding='same')(in1)
        pool1 = MaxPooling2D()(conv1)
        conv2 = Conv2D(1024, (2, 2), padding='valid')(pool1)  # 1024x1x1
        conv3 = Conv2D(512, (1, 1), padding='valid')(conv2)  # 1024x1x1
        conv4 = Conv2D(1, (1, 1), padding='valid')(conv3)
        last = Activation('sigmoid')(Reshape([1])(conv4))
        return Model(inputs=[in1], outputs=[last], name='Discriminator')

    dis_model = DiscriminatorModel()

    fake_input = Input(input_shape, name='fake_input')
    domain_res_fake = Activation(name='dis_fake', activation=None)(dis_model(encoder_model(fake_input)))
    domain_res_real = Activation(name='dis_real', activation=None)(dis_model(encoder_model(inputs)))

    model = Model(inputs=[inputs, fake_input],
                  outputs=[seg_res, domain_res_fake, domain_res_real],
                  name='MainModel')
    model.compile(optimizer=optimizer(lr=initial_learning_rate),
                  loss={
                      'seg_res': loss_function,
                      'dis_fake': 'binary_crossentropy',
                      'dis_real': 'binary_crossentropy'
                  },
                  metrics={
                      'seg_res': metrics,
                      'dis_fake': 'accuracy',
                      'dis_real': 'accuracy'
                  })
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2)):
    up_sample = UpSampling2D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2
