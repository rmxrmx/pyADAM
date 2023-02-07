"""
Script for generating onsets from given onsets in real time.

Setup: set the parameters and models to use.
"""
import time
import numpy as np

# Parameters: Default IOI, timekeeper noise, alpha, beta, delta, gamma
TDEFAULT = 600.0
TVAR = 10
ALPHA = 0.7
BETA = 0.6
PREDICT = 0.5 # = delta
ANTCORR = 0.9 # = gamma = phi

# Keep which models you want to use here.
# If you keep both models, they will be used in the order in which they appear.
# So f.e. iti[0] will contain the ITIs for the first model, while iti[1] will be for the second model.
# If only one model is kept then iti[0] will be for it and iti[1] will be empty.
MODELS = ["adaptation", "jointmodelbeta"]

# instantiating all of the variables needed for the model(s)
ioi = []
tones = []
nt = []
gm = []

T = [[], []]
asyn = [[], []]
adapt = [[], []]
adapt_out = [[], []]
tap = [[], []]
antic = [[], []]
antic_out = [[], []]
iti = [[], []]


# index used in the loop
i = 0

input_time = time.time()

# TODO: this should be replaced by whatever finishing condition
# (i.e. some signal that the experiment is over)
# so it would say -
# while [experiment not finished]:
while i < 42:

    # TODO: wait here until you get an input from the participant and then save it as a IOI
    # not sure how this will operate in ROS, so I just assume I waited for TDEFAULT time here
    time.sleep(TDEFAULT / 1000)

    # get difference between the last time you were here and now to calculate the IOI
    ioi.append((time.time() - input_time) * 1000)
    input_time = time.time()

    if tones:
        tones.append(tones[-1] + ioi[i])
    else:
        tones.append(ioi[i])

    # randomness is introduced to account for noise
    nt.append(np.random.normal(0, TVAR))
    gm.append(np.random.gamma(4, 2.5))

    for j, model in enumerate(MODELS):
        # ADAPTATION MODULE
        # if it is the first two taps
        if i < 2:
            T[j].append(TDEFAULT)
        else:
            T[j].append(T[j][i-1] - BETA * asyn[j][i-1] + nt[i])

        # if it is the first two taps
        if i < 2:
            adapt[j].append(TDEFAULT)
            adapt_out[j].append(tones[i])
        else:
            # in joint_model_beta, alpha is set to 0
            if model =="jointmodelbeta":
                adapt[j].append(T[j][i])
            else:
                adapt[j].append(T[j][i] - ALPHA * asyn[j][i-1])

            adapt_out[j].append(tap[j][i-1] + adapt[j][i])

        # ANTICIPATION MODULE
        # if it is the first two taps
        if i < 2:
            antic[j].append(TDEFAULT)
            antic_out[j].append(tones[i])
        else:
            x = [i-1, i]
            y = [ioi[i-2], ioi[i-1]]

            p = np.polyfit(x, y, 1)

            antic[j].append(PREDICT * np.polyval(p, i+1) + (1-PREDICT) * ioi[i-1] + nt[i])
            antic_out[j].append(tones[i-1] + antic[j][i])


        # JOINT MODULE
        if i < 2:
            tap[j].append(tones[i])
        else:
            # adaptation model does not use the anticipation module, so it is ignored here
            if model == "adaptation":
                tap[j].append(adapt_out[j][i] + gm[i] - gm[i-1])
            else:
                tap[j].append(adapt_out[j][i] - ANTCORR * (adapt_out[j][i] - antic_out[j][i]) + gm[i] - gm[i-1])

        # calculate the asynchronies and the ITIs
        if i < 2:
            asyn[j].append(0)
            iti[j].append(TDEFAULT)
        else:
            asyn[j].append(tap[j][i] - tones[i])
            iti[j].append(tap[j][i] - tap[j][i-1])

        # TODO: The response to the robot should be sent here.
        # iti[0][i] contains the inter tap interval for the robot (using first model).
        # i.e. this is the time it should wait between the previous tap and the next one.

    i += 1

# uncomment these if you want to save the results
# import pandas as pd
# results = pd.DataFrame(data=[iti, asyn, ioi]).transpose()
# results.to_csv(f"interrogator_results_{ALPHA}_{BETA}_{PREDICT}_{ANTCORR}_{MODEL}.csv", index=None, header=None)