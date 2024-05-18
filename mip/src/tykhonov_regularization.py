import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def grad(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the gradient of an image I.
    --------
    Parameters:
    I: np.ndarray
    --------
    Returns:
    grad_x: np.ndarray    
    """
    I = np.array(I)
    grad_x = np.gradient(I, axis=0)
    grad_y = np.gradient(I, axis=1)
    return grad_x, grad_y

def div(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """
    Computes the divergence of the gradient.
    --------
    Parameters:
    grad_x: np.ndarray
    grad_y: np.ndarray
    --------
    Returns:
    div: np.ndarray    
    """
    div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
    return div

def laplacian(I: np.ndarray) -> np.ndarray:
    """
    Computes the laplacian of an image I.
    --------
    Parameters:
    I: np.ndarray
    --------
    Returns:
    lap: np.ndarray
    """
    I = np.array(I)
    lap = np.gradient(np.gradient(I, axis=0), axis=0) + np.gradient(np.gradient(I, axis=1), axis=1)
    return lap