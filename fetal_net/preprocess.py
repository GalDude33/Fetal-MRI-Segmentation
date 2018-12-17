from scipy import ndimage
import numpy as np


def norm_minmax(d):
    return -1 + 2 * (d - d.min()) / (d.max() - d.min())


def laplace(d):
    return ndimage.laplace(d)


def laplace_norm(d):
    return norm_minmax(laplace(d))


def grad(d):
    grads = np.zeros_like(d)
    for a in range(d.squeeze().ndim):
        grads += np.power(ndimage.sobel(d.squeeze(), axis=a), 2)
    return np.sqrt(grads)


def grad_norm(d):
    return norm_minmax(grad(d))
