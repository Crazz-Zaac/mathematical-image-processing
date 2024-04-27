from typing import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2

class ComputeHistogram:
    
    def __init__(self, I: Any, nbins: int) -> None:
        self.nbins = nbins 
        self.image = I
        self.output_hist = None
    
    def my_hist(self):
        img_array = np.array(self.image)
        self.output_hist, bins = np.histogram(img_array.flatten(), bins=self.nbins, range=(0, 255))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        ax1.imshow(self.image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.bar(bins[:-1], self.output_hist, width=3)
        ax2.set_title("Histogram of pixels distribution")
        ax2.set_xlabel("pixel values")
        ax2.set_ylabel("pixel counts")
        
        plt.show()
        

class LinearTransform:
    
    def __init__(self, I: Any, range_min:int, range_max:int) -> None:
        self.image = I
        self.range_min = range_min
        self.range_max = range_max
        self.I_transformed = None
        
    def hist_linear(self):
        image_array = np.array(self.image)
        a = np.min(self.image)
        b = np.max(self.image)
        
        self.I_transformed = ((image_array - a) / (b - a)) * (self.range_max - self.range_min) + self.range_min
        
        return np.clip(self.I_transformed, self.range_min, self.range_max)
        
        
class NonLinearGammaTransform:
    
    def __init__(self, I: Any, gamma: float) -> None:
        self.image = I
        self.gamma = gamma
        self.I_gamma = None
    
    def hist_gamma(self):
        image_array = np.array(self.image)
        self.I_gamma = 255 * np.power(image_array / 255, self.gamma)
        
        return np.clip(self.I_gamma, 0, 255).astype(np.uint8)


class HistogramEqualization:
    """
    It is used to spread the histogram over the entire 
    intensity scale.
    """
    
    
    def __init__(self, I: Any) -> None:
        self.image = I
        self.I_equalized = None
        
    def hist_eq(self):
        image_array = np.array(self.image)
        hist, bins = np.histogram(image_array, bins=1000, range=(0, 255), density=True)
        
        #cumulative sum of histogram
        cdf = hist.cumsum()
        
        # normalized histogram
        normalized_cdf = 255 * cdf / cdf[-1]
        self.I_equalized = np.interp(image_array.flatten(), bins[:-1], normalized_cdf).reshape(image_array.shape)
        

if __name__ == '__main__':
    image = Image.open('goldhill.bmp', 'r')
    # calc_hist = ComputeHistogram(image, nbins=120)
    # calc_hist.my_hist()
    
    hist_eq = HistogramEqualization(image)
    hist_eq.hist_eq()
    plt.imshow(hist_eq.I_equalized, cmap='gray')
    plt.show()
    
    # nlgt = NonLinearGammaTransform(image, 0.2)
    # gamma_transformed = nlgt.hist_gamma()
    # plt.imshow(gamma_transformed, cmap='gray')
    # plt.show()
    
    # lt = LinearTransform(image, 0, 255)
    # linear_transformed = lt.hist_linear()
    # plt.imshow(linear_transformed, cmap='gray')
    # plt.show()
    