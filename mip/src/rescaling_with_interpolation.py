from typing import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2


def tent_interpolation(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

def bell_interpolation(x):
    return np.where(np.abs(x) <= 0.5, -(x ** 2) + 3/4, np.where(np.abs(x) < 1.5, (3/2 - np.abs(x)) ** 2 * 0.5, 0))

def mitchell_netrevalli_interpolation(x):
    return np.where(np.abs(x) < 1, (7/6) * np.abs(x) ** 3 - (7/3) * np.abs(x) ** 2 + 1,
                    np.where(np.logical_and(1 <= np.abs(x), np.abs(x) < 2),
                             (-7/18) * np.abs(x) ** 3 + (7/6) * np.abs(x) ** 2 + (2/3) * np.abs(x) - (10/9), 0))

def resize_filter(image, factor, interpolation):
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    new_height = int(height * factor)
    new_width = int(width * factor)
    
    # Create grid of indices for new image
    y, x = np.mgrid[0:new_height, 0:new_width]
    
    # Calculate corresponding coordinates in original image
    x_orig = (x + 0.5) / factor - 0.5
    y_orig = (y + 0.5) / factor - 0.5
    
    # Calculate indices of surrounding pixels in original image
    x0 = np.floor(x_orig).astype(int)
    x1 = np.minimum(x0 + 1, height - 1)
    # Calculate indices of surrounding pixels in original image
    y0 = np.clip(np.floor(y_orig).astype(int), 0, width - 1)
    y1 = np.clip(np.minimum(y0 + 1, width - 1), 0, width - 1)

    
    # print("y0:", y0)
    # print("y1:", y1)
    
    # Calculate weights for interpolation
    dx = x_orig - x0
    dy = y_orig - y0
    
    # Apply the interpolation filter
    if interpolation == 'tent':
        weight = tent_interpolation
    elif interpolation == 'bell':
        weight = bell_interpolation
    elif interpolation == 'mitchell_netrevalli':
        weight = mitchell_netrevalli_interpolation
    else:
        raise ValueError("Interpolation filter not supported")
    if channels == 1:
        resized_image = (
            weight(dx) * weight(dy) * image[x0, y0] +
            weight(dx) * (1 - weight(dy)) * image[x0, y1] +
            (1 - weight(dx)) * weight(dy) * image[x1, y0] +
            (1 - weight(dx)) * (1 - weight(dy)) * image[x1, y1]
        )
    else:
        resized_image = (
            weight(dx)[:, :, np.newaxis] * weight(dy)[:, :, np.newaxis] * image[x0, y0, :] +
            weight(dx)[:, :, np.newaxis] * (1 - weight(dy))[:, :, np.newaxis] * image[x0, y1, :] +
            (1 - weight(dx))[:, :, np.newaxis] * weight(dy)[:, :, np.newaxis] * image[x1, y0, :] +
            (1 - weight(dx))[:, :, np.newaxis] * (1 - weight(dy))[:, :, np.newaxis] * image[x1, y1, :]
        )
    
    return resized_image.astype(np.uint8)


# img = read(r"../assets/pout.png", cmap='gray')

# f_image = resize_filter(img, 5, 'tent')


# subplot_images([img,  f_image], 
#                titles=['Original Image', 'Filtered image'])