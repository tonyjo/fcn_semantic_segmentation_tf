from __future__ import print_function
from __future__ import division
import os
import numpy as np

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(gt, pred, n_cl):
    hist = fast_hist(gt.flatten(), pred.flatten(), n_cl)

    return hist
