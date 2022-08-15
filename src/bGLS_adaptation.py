import numpy as np
from scipy.stats import multivariate_normal

def bGLS_adaptation(A, b, N, P, ITER, Kratio):
    TRESH = 0.001
    k11 = 2
    k12 = -1
    k13 = 0

    zold = [-9999] * (2 * P)

    for iter in range(ITER):
        cc = np.diag(k11 * np.ones(N), 0) + np.diag(k12 * np.ones(N - 1), 1) + np.diag(k12 * np.ones(N - 1), -1) + np.diag(k13 * np.ones(N - 2), 2) + np.diag(k13 * np.ones(N - 2), -2)
        inv_c = np.linalg.inv(cc)
        z = np.linalg.inv(A @ inv_c @ np.array(A).T) @ A @ inv_c @ b
        d = np.array(A).T @ z - b

        # need to look into covariance - set to N for now, but this is different from MATLAB
        k = np.cov(d[:-1], d[1:], bias=True)

        # k
        k11 = (k[0][0] + k[1][1]) / 2
        k12 = k[0][1]

        # clip bounds the variable between the given boundaries
        k12 = np.clip(k12, (-1 * (4 + Kratio) / (2 * Kratio + 6)) * k11, -0.5 * k11)

        k13 = (k11 + 2 * k12) / -2

        if np.sum(np.abs(z - zold)) < TRESH:
            break

        zold = z

    alpha = z[P : (P * 2)][0]
    beta = -1 * (z[:(P)] + z[P : (P * 2)])[0]
    # different from MATLAB, but more readable
    sM = np.sqrt(k13)
    sT = np.sqrt(2 * k11 + 3 * k12)

    cc = np.diag(k11 * np.ones(N), 0) + np.diag(k12 * np.ones(N - 1), 1) + np.diag(k12 * np.ones(N - 1), -1) + np.diag(k13 * np.ones(N - 2), 2) + np.diag(k13 * np.ones(N - 2), -2)
    d = np.array(A).T @ z - b

    # equivalent to MATLAB's mvpdf
    ll = np.log2(multivariate_normal([0] * len(b), cc).pdf(d))

    return alpha, beta, sM, sT, ll