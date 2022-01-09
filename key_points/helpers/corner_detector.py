#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
from scipy import ndimage

def get_sobel_kernel():
    """The function gives Sobel x and y derivative for corner detection."""
    dx = (1.0 / 8.0) *  np.asarray([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=float)
    dy = (1.0 / 8.0) * np.asarray([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=float)
    return dx, dy

def get_scharr_kernel():
    """The function gives Scharr x and y derivative for corner detection."""
    dx = (1.0/32.0) * np.asarray([[3.0, 10.0, 3.0], [0.0,0.0,0.0], [-3.0, -10.0, -3.0]], dtype=float)
    dy = (1.0/32.0) * np.asarray([[3.0, 0.0, -3.0], [10.0, 0.0,-10.0], [3.0, 0.0, -3.0]], dtype=float)
    return dx, dy

def get_box_kernel(kernel_size):
    """Returns 2D box kernel of the size w."""
    box_kernel = (1.0/ kernel_size**2) * np.ones( (kernel_size, kernel_size)) 
    return box_kernel

def compute_corners(I, type, T, kernel="sobel"):
    """
    Compute corner keypoints using Harris and Shi-Tomasi criteria
    
    Parameters
    ----------
    I : float [MxN] 
        grayscale image

    type :  string
            corner type ('harris' or 'Shi-Tomasi')

    T:  float
        threshold for corner detection
    
    Returns
    -------
    corners : numpy array [num_corners x 2] 
              Coordinates of the detected corners.    
    """
    if kernel == "sobel":
        dx, dy = get_sobel_kernel()
    else:
        dx, dy = get_scharr_kernel()
    Jx = ndimage.convolve(I, dx)
    Jy = ndimage.convolve(I, dy)
    # Get derivatives
    Jxx = Jx**2
    Jyy = Jy**2
    JxJy = Jx*Jy
    box_filter = get_box_kernel(3)
    sum_Jxx = ndimage.convolve(Jxx, box_filter)
    sum_Jyy = ndimage.convolve(Jyy, box_filter)
    sum_JxJy = ndimage.convolve(JxJy, box_filter)
    # Structure matrix
    # M = |sum_Jxx  sum_JxJy|
    #     |sum_JxJy sum_Jyy|
    det_M = sum_Jxx * sum_Jyy - sum_JxJy * sum_JxJy
    trace_M = sum_Jxx + sum_Jyy
    result = None
    if type == 'harris':
        # k in range [0.04, 0.06]
        k = 0.04
        # R == 0.0 - flat; R < 0.0 - edge; R >> 0.0 - corner
        result = R = det_M - k * trace_M ** 2
    if type == 'shi-tomasi':
        result = lambda_min = trace_M / 2.0  -  0.5 * np.sqrt( trace_M** 2 - 4 * det_M)
    if result is not None:
        corners = np.asarray(np.where(result >= T)).T
        print(f"type {type}, corners {corners.shape}")
        return corners
    else:
        raise(ValueError(f"Unknown corner detector: {type}, choose from: 'harris', 'shi-tomasi'!"))

