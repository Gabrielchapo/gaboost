import numpy as np


def sigmoid_derv(s):
    return s * (1 - s)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)