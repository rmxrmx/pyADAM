import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IOI = pd.read_csv('seq2.csv', header=None)[0].tolist()
tones = np.cumsum(IOI)

params = [600, 10, 1, .5, 1, .1]
Tdefault = params[0]
Tvar = params[1]

alpha = params[2]
beta = params[3]
predict = params[4]

antcorr = params[5]

m = len(IOI)

nt = np.zeros(m)
gm = np.zeros(m)
T = np.ones(m) * Tdefault
adapt = np.ones(m) * Tdefault
adapt_out = tones
antic = np.ones(m) * Tdefault
antic_out = tones
tap = tones
iti = np.ones(m) * Tdefault
asyn = np.zeros(m)

for i in range(2, m):
    nt[i] = np.random.normal(0, Tvar)
    gm[i] = np.random.gamma(4, 2.5)

    # ADAPTATION MODULE
    T[i] = T[i-1] - beta * asyn[i-1] + nt[i]

    adapt[i] = T[i] - alpha * asyn[i-1]
    adapt_out[i] = tap[i-1] + adapt[i]

    # ANTICIPATION MODULE
    x = [i-2, i-1]
    y = [IOI[i-2], IOI[i-1]]

    p = np.polyfit(x, y, 1)

    antic[i] = predict * np.polyval(p, i) + (1-predict) * IOI[i-1]
    antic_out[i] = tones[i-1] + antic[i]

    # JOINT MODULE
    tap[i] = adapt_out[i] + antcorr * (adapt_out[i] - antic_out[i]) + gm[i] - gm[i-1]

    asyn[i] = tap[i] - tones[i]
    iti[i] = tap[i] - tap[i-1]

fig, ax = plt.subplots()
ax.plot(IOI)
ax.plot(iti)

plt.show()