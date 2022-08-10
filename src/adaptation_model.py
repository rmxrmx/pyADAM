import numpy as np

from src.bGLS_adaptation import bGLS_adaptation

def adaptation_model(r, e, ITER=20, Kratio=1):

    N = len(r) - 4
    # different from MATLAB - to discuss
    P = 1
    assert(len(r) == len(e))

    e = e - np.mean(e)
    b3 = np.subtract(r[4:], r[3:-1])
    a3 = [e[3:-1], e[2:-2]]

    return bGLS_adaptation(a3, b3, N, P, ITER, Kratio)