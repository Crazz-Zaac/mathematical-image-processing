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
    
    def resize_copy(self) -> np.ndarray:
        image_array = np.array(self.image)
        
        height, width = image_array.shape[:2]
        
        new_height = int(height * self.factor)
        new_width = int(width * self.factor)
        
        # creating a new image array with the new height and width
        resized_image = np.zeros((new_height, new_width, image_array.shape[2]), dtype=np.uint8)
        
        # copying original image pixels to new image pixels
        resized_image[:new_height:self.factor, :new_width:self.factor] = image_array
        
        return resized_image


class RescaleByZeroPadding:
    
    def __init__(self, I: Any, factor: int) -> None:
        self.image = I
        self.factor = factor
        self.output_image = None
    
    def zero_padding(self):
        image_array = np.array(self.image)
        
        height, width = image_array.shape[:2]
        (cX, cY) = (int(height / 2.0), int(width / 2.0))
        
        fft = np.fft.fft2(image_array)
        fft_shifted = np.fft.fftshift(fft)
        
        #zero out the higher frequency values
        fft_shifted[cY - self.factor:cY + self.factor, cX - self.factor:cX + self.factor] = 0
        
        #selecting the Fourier transform
        fft_scaled = fft_shifted * self.factor**2
        
        #selecting the inverse Fourier transform
        ifft = np.fft.ifft2(np.fft.ifftshift(fft_scaled))
        
        # taking the real part of image and clipping values to [0, 255]
        self.output_image = np.clip(np.abs(ifft), 0, 255).astype(np.uint8)
        
        return self.output_image
        

class RescaleWithInterpolation:
    
    def __init__(self, I: Any, factor: int, filter: str) -> None:
        self.image = I
        self.factor = factor
        self.filter = filter
        self.output_image = None
    
    def resize_filter(self):
        image_array = np.array(self.image)
        
        if self.filter == 'tent':
            x = np.linspace(-1, 1, num=self.factor)
            h_x = np.array([(1 - np.abs(x)) if abs(x) <= 1 else 0 for x in x])
        
        elif self.filter == 'bell':
            x = np.linspace()
            




if __name__ == '__main__':
    
    image = Image.open('goldhill.bmp', 'r')
    # rescale = RescaleByDuplication(image,5)
    # rescaled_image = rescale.resize_copy()
    # plt.imshow(rescaled_image, cmap='gray')
    # plt.show()
    
    # resize_image = RescaleByDuplication(image, 2)
    # I_resize = resize_image.resize_copy()
    # print("Size of original image: " + str(image.size))
    # print("Size of resized image: " + str(I_resize.shape))
    # plt.imshow(I_resize, cmap='gray')
    # plt.show()
    
    ## zero padding with FFT
    zero_padding = RescaleByZeroPadding(image, 2)
    I_zeropad = zero_padding.zero_padding()
    plt.imshow(I_zeropad, cmap='gray')
    plt.show()