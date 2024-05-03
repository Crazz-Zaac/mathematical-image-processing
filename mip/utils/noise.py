import numpy as np
from typing import Tuple


def gaussian_noise(mean: float, std: float, size: Tuple[int, int]) -> np.ndarray:
    """
    Generate Gaussian noise with a given mean and standard deviation.

    Parameters:
    -----------
    * mean: float
        Mean of the Gaussian noise.
    * std: float
        Standard deviation of the Gaussian noise.
    * size: Tuple[int, int]
        Size of the noise.

    Returns:
    --------
    * np.ndarray
        Gaussian noise with the given mean and standard deviation.
    """
    return np.random.normal(mean, std, size)


def salt_pepper_noise(salt: float, pepper: float, size: Tuple[int, int]) -> np.ndarray:
    """
    Generate salt and pepper noise with a given salt and pepper ratio.

    Parameters:
    -----------
    * salt: float
        Salt ratio of the noise.
    * pepper: float
        Pepper ratio of the noise.
    * size: Tuple[int, int]
        Size of the noise.

    Returns:
    --------
    * np.ndarray
        Salt and pepper noise with the given salt and pepper ratio.
    """
    noise = np.random.rand(*size)
    salt_mask = noise < salt
    pepper_mask = noise > 1 - pepper
    salt_pepper_mask = salt_mask + pepper_mask
    return salt_pepper_mask


def poisson_noise(lam: float, size: Tuple[int, int]) -> np.ndarray:
    """
    Generate Poisson noise with a given lambda.

    Parameters:
    -----------
    * lam: float
        Lambda of the Poisson noise.
    * size: Tuple[int, int]
        Size of the noise.

    Returns:
    --------
    * np.ndarray
        Poisson noise with the given lambda.
    """
    return np.random.poisson(lam, size)
