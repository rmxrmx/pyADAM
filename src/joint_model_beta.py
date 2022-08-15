import numpy as np
from src.bGLS_joint_beta import bGLS_joint_beta

def joint_model_beta(s, r, e, L, H, ITER=20):
    e = e - np.mean(e)

    # b - Ax
    b = r[4:]
    A = [None, None, None]
    A[0] = np.subtract(s[3:-1], s[2:-2])
    A[1] = np.subtract(s[3:-1], e[3:-1])
    A[2] = -1 * np.cumsum(e[3:-1])

    # correct for mean bGLS

    bm = b - np.mean(b)
    Am = A
    Am[0] = A[0] - np.mean(A[0])
    Am[1] = A[1] - np.mean(A[1])
    # The third vector is commented out in matlab code

    # do bGLS with lower / higher bounds
    xB, sMB, sTB, LLE = bGLS_joint_beta(Am, bm, L, H, ITER)

    # calculate parameter estimates
    gammaE = 1 - xB[1]
    mE = xB[0] / (1 - gammaE)
    betaE = xB[2] / gammaE
    stE = sTB / np.sqrt(1 + 2 * (gammaE**2) - 2 * gammaE)

    smE = sMB

    return gammaE, mE, betaE, stE, smE, LLE