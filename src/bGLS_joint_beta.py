import numpy as np
from scipy.stats import multivariate_normal

def bGLS_joint_beta(A, b, L, H, ITER):
    assert(len(A) == 3)

    # A and b must have 0 mean
    # normal regression
    z = np.linalg.lstsq(np.array(A).T, b, rcond=None)[0]
    z2 = z

    # Iterate to find the maximum likelihood estimator

    for i in range(ITER):
        # calculate the residual noise covariance
        d = np.array(A).T @ z2 - b

        # need to look into covariance - set to N for now, but this is different from MATLAB
        k = np.cov(d[:-1], d[1:], bias=True)

        # k
        k11 = (k[0][0] + k[1][1]) / 2

        k12 = k[0][1]

        if k12 > 0:
            sM = 0
        else:
            sM = np.sqrt(-1 * k12)
        
        if k11 < 3 * (sM**2):
            sM = np.sqrt(k11 / 3)

        # motor variance gets killed here
        sM = 0

        sT = np.sqrt(k11 - sM**2)

        k11 = (sT**2) + 2 * (sM**2)
        k12 = -1 * (sM**2)

        # calculate GLS with known covariance
        nn = len(b)

        cc = np.diag(k11 * np.ones(nn), 0) + np.diag(k12 * np.ones(nn - 1), 1)
        inv_c = np.linalg.inv(cc)

        z2 = np.linalg.inv(A @ inv_c @ np.array(A).T) @ A @ inv_c @ b
        
        # clip bounds the variable between the given boundaries
        z2[1] = np.clip(z2[1], 0.1, 0.9)

        if z2[1] > 0:
            z2[0] = np.clip(z2[0], L * z2[1], H * z2[1])
        else:
            z2[0] = np.clip(z2[0], H * z2[1], L * z2[1])


    x = z2

    # equivalent to MATLAB's mvpdf
    ll = np.log2(multivariate_normal([0] * len(b), cc).pdf(x @ A - b))

    return x, sM, sT, ll