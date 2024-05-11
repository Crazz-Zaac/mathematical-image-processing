
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional


def read(path: str, cmap: str) -> np.ndarray:
    """
    Read an image using OpenCV.

    CMAP: RGB, BGR, GRAY
    """
    image = cv2.imread(path)
    cmap = cmap.upper()
    if cmap == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif cmap == 'GRAY':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif cmap == 'BGR':
        pass
    return image


def subplot_images(images: List[np.ndarray],
                   titles: Optional[List[str]], fig_size: tuple = (10, 10),
                   cmap: str = 'gray', order: Tuple[int, int] = (1, -1)):
    """
    Subplot multiple images using matplotlib.

    """
    order = (order[0], len(images) // order[0]) if order[1] == -1 else \
        (len(images)//order[1], order[1]) if order[0] == -1 else order

    fig, axes = plt.subplots(order[0], order[1], figsize=fig_size)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap=cmap)
            ax.axis('off')
            if titles:
                ax.set_title(titles[i])
            else:
                ax.set_title(f'Image {i+1}')

    plt.show()


def show(image: np.ndarray,
         fig_size: tuple = (10, 10), cmap: str = 'gray', title: str = None):
    """
    Display an image using matplotlib.
    """
    plt.figure(figsize=fig_size)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()
