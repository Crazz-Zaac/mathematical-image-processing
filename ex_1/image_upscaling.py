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

class RescaleByDuplication:
    
    def __init__(self, I: Any, factor: int) -> None:
        self.image = I
        self.factor = factor
        self.output_image = None
    
    def resize_copy(self):
        image_array = np.array(self.image)
        
        height, width = image_array.shape[:2]
        
        new_height = int(height * self.factor)
        new_width = int(width * self.factor)
        
        # creating a new image array with the new height and width
        resized_image = np.zeros((new_height, new_width, image_array.shape[2]), dtype=np.uint8)
        
        # copying original image pixels to new image pixels
        resized_image[:new_height:self.factor, :new_width:self.factor] = image_array
        
        return resized_image


if __name__ == '__main__':
    
    image = Image.open('goldhill.bmp', 'r')
    # rescale = RescaleByDuplication(image,5)
    # rescaled_image = rescale.resize_copy()
    # plt.imshow(rescaled_image, cmap='gray')
    # plt.show()
    
    resize_image = RescaleByDuplication(image, 2)
    I_resize = resize_image.resize_copy()
    plt.imshow(I_resize, cmap='gray')
    plt.show()