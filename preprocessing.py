import numpy as np
from PIL import Image
from scipy import ndimage

def gaussian_filter(im: np.ndarray, window: int, sigma) -> np.ndarray:
    '''
    Use scipy's gaussian_filter method to apply a gaussian filter.
    The truncate value has to be calculated to get scipy to use a specific window size.
    See: https://stackoverflow.com/a/43617491
    '''
    truncate = (((window - 1) / 2) - 0.5) / sigma
    return ndimage.gaussian_filter(im, sigma=sigma, truncate=truncate)

def blur_image(im: np.ndarray, window = 7, sigma = 1.5) -> np.ndarray:
    '''
    Blur RGB image with a low-pass Gaussian filter.
    '''
    return gaussian_filter(im, window=7, sigma=1.5)

def brightness_equalization(im: np.ndarray, window=5, sigma=0.5) -> np.ndarray:
    '''
    Perform brightness equalization according to Zhang et al
    '''
    # Getting grayscale: https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale
    grayscale = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    grayscale_3d = np.stack([grayscale] * 3, axis=2) + 0.0001
    return im / gaussian_filter(grayscale_3d, window=window, sigma=sigma)

def preprocess_zhang(im: np.ndarray, normalize=False) -> np.ndarray:
    im_blur = blur_image(im, window=7, sigma=1.5)
    im_eq = brightness_equalization(im_blur, window=5, sigma=0.5)
    if normalize is True:
        return im_eq / np.amax(im_eq)
    else:
        return im_eq
