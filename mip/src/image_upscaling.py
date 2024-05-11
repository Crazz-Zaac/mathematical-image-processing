from typing import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2

"""
The goal of this exercise is to manipulate image grids and pixel coordinates by upscaling
a given image.
"""


def resize_copy(image_array: np.ndarray, factor: int) -> np.ndarray:
    """
    This function uses np.kron multiplier to duplicate the pixels in the image array.

    Args:
    -----
    image_array: np.ndarray
        The image array to be resized.
    factor: int
        The factor by which the image is to be resized.

    Returns:
    --------
    resized_image: np.ndarray
        The resized image array.


    Usage of np.kron:
    np.kron(np.arange(0, 16).reshape(4, 4), np.ones((2,2)))
    array([[ 0.,  0.,  1.,  1.,  2.,  2.,  3.,  3.],
            [ 0.,  0.,  1.,  1.,  2.,  2.,  3.,  3.],
            [ 4.,  4.,  5.,  5.,  6.,  6.,  7.,  7.],
            [ 4.,  4.,  5.,  5.,  6.,  6.,  7.,  7.],
            [ 8.,  8.,  9.,  9., 10., 10., 11., 11.],
            [ 8.,  8.,  9.,  9., 10., 10., 11., 11.],
            [12., 12., 13., 13., 14., 14., 15., 15.],
            [12., 12., 13., 13., 14., 14., 15., 15.]])

    """

    resized_image = np.kron(image_array, np.ones((factor, factor)))

    return resized_image


def zero_padding(image_array: np.ndarray, factor: int) -> np.ndarray:
    """
    Perform zero padding on the Fourier transform of the image array.

    Args:
    -----
    image_array: np.ndarray
        The image array to be zero padded.
    factor: int
        The factor by which the image is to be zero padded.

    Returns:
    --------
    I_padded: np.ndarray
        The zero padded image array.

    """
    height, width = image_array.shape

    # Compute the Fourier transform of the image array.
    fft_img = np.fft.fft2(image_array)

    # shifted fft for shifting the zero frequency component to the center of the spectrum
    fft_shifted = np.fft.fftshift(fft_img)

    # Zero pad the Fourier transform.
    row_pad = np.zeros((1, width))
    column_pad = np.zeros((height+2, 1))

    padded_image = np.vstack([row_pad, fft_shifted, row_pad])

    padded_image = np.hstack([column_pad, padded_image, column_pad])

    padded_image = np.fft.ifftshift(padded_image)

    # Compute the inverse Fourier transform of the zero padded Fourier transform.
    I_padded = np.fft.ifft2(padded_image)

    # multiply by the factor to get the correct intensity values
    I_padded = I_padded * factor

    return I_padded.real.astype(np.uint8)
