import numpy as np


def adam_single(theta, s):

    # % Implements the Adaptation and Anticipation model of sensorimotor
    # % synchronisation.

    # % INPUT ARGS - theta: vector of input parameters for the simulated agent
    # %                  theta(1) -> alpha, phase correction
    # %                  theta(2) -> beta, period correction
    # %                  theta(3) -> delta, predicition-tracking weight
    # %                  theta(4) -> gamma, anticipatory correction
    # %                  theta(5) -> Tvar, timekeeper variance
    # %
    # %            - S: vector of stimulus onset times from the initial tone i=0
    # %
    # % OUTPUT - R: vector of response onsets
    # %
    # %        - r: vector of inter-response intervals
    # %             r(i) = R(i) - R(i-1)
    # %
    # %        - e: vector of asynchronies between stimuli and responses
    # %             e = R - S
    # %
    # % NB: First response is produced at i = 3

    # N.B.: some variables were renamed to conform to the snake_case naming style
    # (which is used by the Python community).
    # t_var = Tvar, length = l, stimulus = S, timekeeper = T, norm = N,
    # mo = M, ro = R, err = e, pred = p, track = t, resp = r

    alpha, beta, delta, gamma, t_var = theta

    length = len(s)
    stimulus = np.cumsum(s)

    timekeeper = [0] * length
    timekeeper[:3] = s[:3]

    norm = np.random.normal(0, t_var[0], length)

    mo = [0] * length

    ro = [0] * length
    ro[:3] = stimulus[:3]

    for i in range(2, length - 1):
        # ADAPTATION MODULE
        # calculate error
        err = ro[i] - stimulus[i]

        # perform period correction
        timekeeper[i] = timekeeper[i - 1] - beta * err

        # perform phase correction and set onset time for next response
        r_prime = ro[i] + timekeeper[i] - alpha * err + norm[i]

        # ANTICIPATION MODULE
        # calculate predicted inter-stimulus interval
        pred = 2 * s[i] - s[i - 1]

        # calculate tracking inter-stimulus interval
        track = s[i]

        # weigh predictive and tracking strategies and determine
        # predicted stimulus onset
        s_prime = stimulus[i] + delta * pred + (1 - delta) * track + norm[i]

        # JOINT MODULE

        # calculate predicted error between planned response
        # onset and predicted stimulus onset
        e_prime = r_prime - s_prime

        # perform predictive error correction and generate response
        ro[i + 1] = r_prime - gamma * e_prime + mo[i + 1] - mo[i]

    # CALCULATE OUTPUT
    err = ro - stimulus
    resp = (timekeeper, ro[1:] - ro[:-1])

    return ro, resp, err
