from keras import backend as K
import numpy as np

def gram_matrix(x):
    #assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        pass
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    features = K.reshape(x, [-1, x.shape[1], np.prod(x.shape[2:])])
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(target, style_combination):
    style, combination = style_combination[..., 0], style_combination[..., 1]
    #assert K.ndim(style) == 3
    #assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = int(style.shape[1])
    size = int(style.shape[2] * style.shape[3])
    return K.sum(K.square(S - C), axis=[1,2]) / (4.0 * (channels ** 2) * (size ** 2))