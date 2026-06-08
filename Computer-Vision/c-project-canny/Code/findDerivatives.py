'''
  File name: findDerivatives.py
  Author: Tarmily Wen
  Date created: Dec. 8, 2019
'''

import numpy as np
from scipy import signal
import utils
from PIL import Image
import matplotlib.pyplot as plt

'''
  File clarification:
    Compute gradient put ginformation of the inrayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the derivatives along x-axis
    - Output Magy: H x W matrix represents the derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''

def findDerivatives(I_gray):
    # smoothing kernels
    gaussian = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159.0

    # kernel for x and y gradient
    dx = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    dy = np.asarray([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    ###############################################################################
    # Smooth the image, then convolve with Sobel kernels to get x/y gradients.
    # Magnitude = sqrt(Magx^2 + Magy^2); orientation = arctan2(Magy, Magx).
    ###############################################################################
    I_smooth = signal.convolve2d(I_gray, gaussian, mode='same', boundary='symm')

    Magx = signal.convolve2d(I_smooth, dx, mode='same', boundary='symm')
    Magy = signal.convolve2d(I_smooth, dy, mode='same', boundary='symm')

    Mag = np.sqrt(Magx ** 2 + Magy ** 2)
    Ori = np.arctan2(Magy, Magx)

    return Mag, Magx, Magy, Ori