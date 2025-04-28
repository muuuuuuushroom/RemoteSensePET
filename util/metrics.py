import numpy as np


def compute_relerr(gt: np.ndarray, pd: np.ndarray):
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        rmae = np.mean(np.abs(diff) / gt) * 100
        rmse = np.sqrt(np.mean(diff**2 / gt**2)) * 100
    else:
        rmae = 0
        rmse = 0
    return rmae, rmse


def compute_racc(gt: np.ndarray, pd: np.ndarray):
    err = sum(abs(gt-pd)) / gt.sum()
    racc = 1 - err
    
    return racc