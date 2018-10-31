from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np


def get_main_connected_component(data):
    labeled_array, num_features = label(data)
    i = np.argmax([np.sum(labeled_array == _) for _ in range(1, num_features + 1)]) + 1
    return labeled_array == i


def postprocess_prediction(pred, gaussian_std=1, threshold=0.5, fill_holes=True, connected_component=True):
    pred = gaussian_filter(pred, gaussian_std) > threshold
    if fill_holes:
        pred = binary_fill_holes(pred)
    if connected_component:
        pred = get_main_connected_component(pred)
    return pred
