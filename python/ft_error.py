import numpy as np


def cross_entropy(Y_pred, Y_real):
    diff = Y_pred - Y_real
    return diff / Y_real.shape[0]

def error(Y_pred, Y_real):
    logged = - np.log(Y_pred[np.arange(Y_real.shape[0]), Y_real.argmax(axis=1)])
    return np.sum(logged) / Y_real.shape[0]
