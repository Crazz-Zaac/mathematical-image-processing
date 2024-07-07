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
    if cmap == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif cmap == "GRAY":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif cmap == "BGR":
        pass
    return image


def subplot_images(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    fig_size: tuple = (20, 10),
    dpi: int = 100,
    cmap: str = "gray",
    order: Tuple[int, int] = (1, -1),
) -> plt.Figure:
    """
    Subplot multiple images using matplotlib.

    Parameters:
        images (List[np.ndarray]): List of images to be plotted.
        titles (Optional[List[str]]): List of titles for the images. Default is None.
        fig_size (tuple): Figure size for the plot. Default is (20, 10).
        dpi (int): Dots per inch for the plot. Default is 100.
        cmap (str): Colormap to be used for the images. Default is 'gray'.
        order (Tuple[int, int]): Order of subplots (rows, columns). Use -1 to auto-calculate one dimension.

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    # Calculate subplot grid dimensions if one is set to -1
    if order[1] == -1:
        order = (order[0], (len(images) + order[0] - 1) // order[0])
    elif order[0] == -1:
        order = ((len(images) + order[1] - 1) // order[1], order[1])

    fig, axes = plt.subplots(order[0], order[1], figsize=fig_size, dpi=dpi)
    plt.tight_layout()

    # Flatten axes array for easy iteration, handling the case where axes is not 2D
    if order[0] == 1 or order[1] == 1:
        axes = np.atleast_1d(axes).flatten()
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap=cmap)
            ax.axis("off")
            if titles:
                ax.set_title(titles[i], fontsize=12)
        else:
            ax.axis("off")  # Turn off axes for empty subplots

    plt.show()
    return fig


def show(
    image: np.ndarray, fig_size: tuple = (10, 10), cmap: str = "gray", title: str = None
):
    """
    Display an image using matplotlib.
    """
    plt.figure(figsize=fig_size)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
