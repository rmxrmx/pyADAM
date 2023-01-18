"""
Script for generating onsets from given onsets.
"""

import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in a list of IOIs that is used as the "other" participant
IOI = pd.read_csv('seq2.csv', header=None)[0].tolist()

# turn the IOIs into onsets
tones = np.cumsum(IOI).astype(float)

# parameters: Default IOI, timekeeper noise, alphas, betas, deltas, gammas
Tdefault = 600.0
Tvar = 10
alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
# alphas = [0.7]
betas = [0, 0.2, 0.4, 0.6, 0.8, 1]
# betas = [0.6]
predicts = [0, 0.5, 1]
# predicts = [0.5]
antcorrs = [0.1, 0.5, 0.9]
# antcorrs = [0.9]

# set how many runs of identical parameters we want to run (primarily used for parameter recovery)
number_of_runs = 100

shutil.rmtree("new_interr_data")
os.mkdir("new_interr_data")
# run through every combination of parameters
# for real-time onset generation, this is not needed
for alpha in alphas:
    for beta in betas:
        for predict in predicts:
            for antcorr in antcorrs:

                for run in range(number_of_runs):

                    # this was used when I generated different data dependant on the model --
                    # if using this script just for a general generation, remove this part
                    models = ["jointmodelbeta", "adaptation"]

                    for model in models:

                        # instantiating all of the variables needed for the model
                        m = len(IOI)

                        nt = np.zeros(m)
                        gm = np.zeros(m)
                        T = np.ones(m).round() * Tdefault
                        adapt = np.ones(m).round() * Tdefault
                        adapt_out = tones.copy()
                        antic = np.ones(m).round() * Tdefault
                        antic_out = tones.copy()    
                        tap = tones.copy()
                        iti = np.ones(m).round() * Tdefault
                        ioi = np.ones(m).round() * Tdefault
                        asyn = np.zeros(m)

                        # only start calculations from the 3rd onset
                        for i in range(2, m):

                            # randomness is introduced to account for noise
                            nt[i] = np.random.normal(0, Tvar)
                            # gm[i] = np.random.gamma(4, 2.5)
                            gm[i] = 0

                            # ADAPTATION MODULE
                            T[i] = T[i-1] - beta * asyn[i-1] + nt[i]

                            # in joint_model_beta, alpha is set to 0
                            # keep only the else part if using script to generate general data
                            if model =="jointmodelbeta":
                                adapt[i] = T[i]
                            else:
                                adapt[i] = T[i] - alpha * asyn[i-1]

                            adapt_out[i] = tap[i-1] + adapt[i]

                            # ANTICIPATION MODULE
                            x = [i-1, i]
                            y = [IOI[i-2], IOI[i-1]]

                            p = np.polyfit(x, y, 1)

                            antic[i] = predict * np.polyval(p, i+1) + (1-predict) * IOI[i-1] + nt[i]
                            antic_out[i] = tones[i-1] + antic[i]


                            # JOINT MODULE
                            # adaptation model does not use the anticipation module, so it is ignored here
                            if model == "adaptation":
                                tap[i] = adapt_out[i] + gm[i] - gm[i-1]
                            else:
                                tap[i] = adapt_out[i] - antcorr * (adapt_out[i] - antic_out[i]) + gm[i] - gm[i-1]

                            # calculate the asynchronies and the ITIs
                            asyn[i] = tap[i] - tones[i]
                            iti[i] = tap[i] - tap[i-1]

                        # uncomment these lines if you want to see the plots for the ITIs and IOIs
                        # fig, ax = plt.subplots()
                        # ax.plot(IOI, label="IOI")
                        # ax.plot(iti, label="ITI (generated)")
                        # ax.set_title(f"Parameter estimations with phi = {antcorr}")
                        # ax.set_xlabel("datapoint")
                        # ax.set_ylabel("IOI / ITI")
                        # ax.legend()
                        # plt.show()
                        # print(IOI)

                        # TAP: onsets1, TONES: onsets2
                        results = pd.DataFrame(data=[iti, asyn, IOI]).transpose()
                        results.to_csv(f"new_interr_data/interrogator_results_{alpha}_{beta}_{predict}_{antcorr}_run{run}_{model}.csv", index=None, header=None)
                        