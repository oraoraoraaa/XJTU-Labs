'''
  File name: utils.py
  Author: Haoyuan(Steve) Zhang
  Date created: 9/10/2017
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def visDerivatives(I_gray, Mag, Magx, Magy):
    fig, (Ax0, Ax1, Ax2, Ax3) = plt.subplots(1, 4, figsize=(20, 8))
    Ax0.imshow(Mag, cmap="gray", interpolation="nearest")
    Ax0.axis("off")
    Ax0.set_title("Gradient Magnitude")
    Ax1.imshow(Magx, cmap="gray", interpolation="nearest")
    Ax1.axis("off")
    Ax1.set_title("Gradient Magnitude (x axis)")
    Ax2.imshow(Magy, cmap="gray", interpolation="nearest")
    Ax2.axis("off")
    Ax2.set_title("Gradient Magnitude (y axis)")
    Mag_vec = Mag.transpose().reshape(1, Mag.shape[0] * Mag.shape[1])
    hist, bin_edge = np.histogram(Mag_vec.transpose(), 100)
    ind_array = np.array(np.where(np.cumsum(hist).astype(float) / hist.sum() < 0.95))
    thr = bin_edge[ind_array[(0, -1)]]
    ind_remove = np.where(np.abs(Mag) < thr)
    Magx[ind_remove] = 0
    Magy[ind_remove] = 0
    X, Y = np.meshgrid(np.arange(0, Mag.shape[1], 1), np.arange(0, Mag.shape[0], 1))
    Ax3.imshow(I_gray, cmap="gray", interpolation="nearest")
    Ax3.axis("off")
    Ax3.set_title("Gradient Orientation")
    Q = plt.quiver(X, Y, Magx, Magy)
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r"$2 \frac{m}{s}$", labelpos="E", coordinates="figure")


def visCannyEdge(Im_raw, M, E):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 12))
    ax0.imshow(Im_raw)
    ax0.axis("off")
    ax0.set_title("Raw image")
    ax1.imshow(M, cmap="gray", interpolation="nearest")
    ax1.axis("off")
    ax1.set_title("Non-Max Suppression Result")
    ax2.imshow(E, cmap="gray", interpolation="nearest")
    ax2.axis("off")
    ax2.set_title("Canny Edge Detection")


def GaussianPDF_1D(mu, sigma, length):
    half_len = length / 2
    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)
    ax = ax.reshape([-1, ax.size])
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp(-np.square(ax - mu) / (2 * sigma * sigma))
    return nominator / denominator


def GaussianPDF_2D(mu, sigma, row, col):
    g_row = GaussianPDF_1D(mu, sigma, row)
    g_col = GaussianPDF_1D(mu, sigma, col).transpose()
    return signal.convolve2d(g_row, g_col, "full")


def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.587 * g + 0.114 * b
    return I_gray
