# -*- coding: utf-8 -*-
import numpy as np


def dotFeature(x, _settings=None):
    """Dots are already 2D features; return as float array."""
    return np.asarray(x, dtype=np.float64), {"type": "identity"}


def timeSeriesFeature(x, _settings=None):
    """Extract compact statistical features from each time series."""
    x = np.asarray(x, dtype=np.float64)

    mean = x.mean(axis=1)
    std = x.std(axis=1)
    min_v = x.min(axis=1)
    max_v = x.max(axis=1)

    dx = np.diff(x, axis=1)
    slope_mean = dx.mean(axis=1)
    slope_std = dx.std(axis=1)

    feats = np.stack([mean, std, min_v, max_v, slope_mean, slope_std], axis=1)
    return feats, {"type": "stats6"}
