import numpy as np

from src.bGLS_adaptation import bGLS_adaptation


def adaptation_model(resp, err, iterations=20, k_ratio=1):

    num = len(resp) - 4
    # different from MATLAB - to discuss
    cap_p = 1
    assert len(resp) == len(err)

    err = err - np.mean(err)
    b_3 = np.subtract(resp[4:], resp[3:-1])
    a_3 = [err[3:-1], err[2:-2]]

    return bGLS_adaptation(a_3, b_3, num, cap_p, iterations, k_ratio)
