import numpy as np


def image_filter(I: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply a filter to an image.

    Parameters:
    -----------
    * I: np.ndarray
        Image to be filtered.  
    * filter: np.ndarray
        Filter to be applied to the image.

    Returns:
    -------
    * np.ndarray
        Filtered image.
    """
    image_array = I
    filter_array = filter
    filter_size = filter_array.shape
    image_size = image_array.shape
    out_size = (image_size[0] - filter_size[0] + 1,
                image_size[1] - filter_size[1] + 1)
    print(f"Original Image Size: {image_size}, Filtered Size: {out_size}")

    image_filtered = np.zeros(out_size)

    for i in range(out_size[0]):
        for j in range(out_size[1]):
            image_filtered[i, j] = np.sum(
                image_array[i:i+filter_size[0], j:j+filter_size[1]] * filter_array)
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    return image_filtered


def median_filter(I: np.array, size: int) -> np.ndarray:
    """
    Apply a median filter to an image.
    Finds median on a window and replaces the pixel value with the median.

    Parameters:
    -----------
    * I: np.ndarray
        Image to be filtered.
    * size: int
        Size of the filter.

    Returns:
    --------
    * np.ndarray
        Filtered image.

    """
    image_array = I
    image_size = image_array.shape
    out_size = (image_size[0] - size + 1,
                image_size[1] - size + 1)
    print(f"Original Image Size: {image_size}, Filtered Size: {out_size}")

    image_filtered = np.zeros(out_size)

    for i in range(out_size[0]):
        for j in range(out_size[1]):
            image_filtered[i, j] = np.median(
                image_array[i:i+size, j:j+size])
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    return image_filtered
