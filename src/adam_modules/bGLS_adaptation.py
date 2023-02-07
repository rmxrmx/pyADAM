import numpy as np
from scipy.stats import multivariate_normal


def bGLS_adaptation(a_3, b_3, num, cap_p, iterations, k_ratio):
    tresh = 0.001
    k11 = 2
    k12 = -1
    k13 = 0

    zold = [-9999] * (2 * cap_p)

    for iteration in range(iterations):
        c_c = (
            np.diag(k11 * np.ones(num), 0)
            + np.diag(k12 * np.ones(num - 1), 1)
            + np.diag(k12 * np.ones(num - 1), -1)
            + np.diag(k13 * np.ones(num - 2), 2)
            + np.diag(k13 * np.ones(num - 2), -2)
        )
        inv_c = np.linalg.inv(c_c)
        z_1 = np.linalg.inv(a_3 @ inv_c @ np.array(a_3).T) @ a_3 @ inv_c @ b_3
        d_1 = np.array(a_3).T @ z_1 - b_3

        # need to look into covariance - set to N for now, but this is different from MATLAB
        k = np.cov(d_1[:-1], d_1[1:], bias=True)

        # k
        k11 = (k[0][0] + k[1][1]) / 2
        k12 = k[0][1]

        # clip bounds the variable between the given boundaries
        k12 = np.clip(k12, (-1 * (4 + k_ratio) / (2 * k_ratio + 6)) * k11, -0.5 * k11)

        k13 = (k11 + 2 * k12) / -2

        if np.sum(np.abs(z_1 - zold)) < tresh:
            break

        zold = z_1

    alpha = z_1[cap_p : (cap_p * 2)][0]
    beta = -1 * (z_1[:(cap_p)] + z_1[cap_p : (cap_p * 2)])[0]
    # different from MATLAB, but more readable
    s_m = np.sqrt(k13)
    s_t = np.sqrt(2 * k11 + 3 * k12)

    c_c = (
        np.diag(k11 * np.ones(num), 0)
        + np.diag(k12 * np.ones(num - 1), 1)
        + np.diag(k12 * np.ones(num - 1), -1)
        + np.diag(k13 * np.ones(num - 2), 2)
        + np.diag(k13 * np.ones(num - 2), -2)
    )
    d_1 = np.array(a_3).T @ z_1 - b_3

    # equivalent to MATLAB's mvpdf
    # loglik = np.log2(multivariate_normal([0] * len(b_3), c_c).pdf(d_1))

    # N.B.: LL set to 0 as it was producing crashes
    loglik = 0

    return alpha, beta, s_m, s_t, loglik
