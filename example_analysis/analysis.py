"""
An example analysis script on some generated SMS data.

Setup: set the data location in [files] and the models you want to use in [models].
Optionally, the results file can be changed at the end of the script.

This script assumes that the data is in two columns, one for each participant, containing onsets.
"""
import glob
import numpy as np
import pandas as pd
from src.adam_modules.joint_model_beta import joint_model_beta
from src.adam_modules.adaptation_model import adaptation_model
from src.qa_data import convert_to_intervals, interpolate_onsets
from src.utils import convert_to_participant

# This folder should contain all of the files that are to be analysed.
files = glob.glob("example_data/*.csv")

# For the purposes of an example, both the models are included,
# But the example data was generated using only joint model beta,
# So the adaptation estimations will not make much sense.
models = ["joint_model_beta", "adaptation"]

# A results dataframe for storing behavioural data and parameter estimations
results = pd.DataFrame(
    columns=[
        "filename",
        "leader",
        "follower",
        "model",
        "alpha",
        "beta",
        "delta",
        "phi",
        "sT",
        "sM",
        "LLE",
        "median_abs_async",
        "min_abs_async",
        "max_abs_async",
        "mean_async",
        "sd_async",
        "mean_abs_async",
        "sd_abs_async",
        "min_async",
        "max_async",
        "median_ITI",
        "mean_ITI",
        "sd_ITI",
        "min_ITI",
        "max_ITI",
        "median_IOI",
        "mean_IOI",
        "sd_IOI",
        "min_IOI",
        "max_IOI",
        "CV_ITI",
        "CV_async",
        "n_events",
        "n_interpolated",
    ]
)

for file in files:
    print(file)
    data = pd.read_csv(file)

    # This section interpolates the onsets and calculates the ITI, IOI and async from them.
    # Note that you can add more participants here if you had them, but this example script
    # is made for 2 participants.

    # N.B.: this is only needed because we have onsets as input.
    # If we have ITIs and IOIs, this is not needed.
    # Note that we have left one of the data files intentionally corrupted, to
    # showcase how this function interpolates. You can find the corresponding
    # result via sorting by n_interpolations in example_results.csv.
    onsets1, n_interpolated1, removals1 = interpolate_onsets(data.iloc[:, 0].tolist())
    onsets2, n_interpolated2, removals2 = interpolate_onsets(data.iloc[:, 1].tolist())

    # If you have more participants, you should add their variables in these lists.
    onsets = [onsets1, onsets2]
    n_interpolated = [n_interpolated1, n_interpolated2]
    removals = [removals1, removals2]

    # Permute through all possible combinations of leader+follower
    # Where 0 = participant1, 1 = participant2
    # For more participants, you can try out the below line of code instead
    # (replace 3 by number of participants)
    # groups = itertools.permutations(range(3), 2)
    groups = [[1, 0], [0, 1]]

    for grouping in groups:
        leader, follower = grouping

        # Converts the onsets into ITIs, IOIs and calculates asynchronies.
        iti, ioi, asyn = convert_to_intervals(onsets[follower], onsets[leader], removals[follower], removals[leader])
        n_interpolations = n_interpolated[leader] + n_interpolated[follower]

        # Convert these to string values for improved readability in the results.
        leader = convert_to_participant(leader)
        follower = convert_to_participant(follower)

        # Round for improved readability.
        iti = np.round(iti, 3)
        ioi = np.round(ioi, 3)
        asyn = np.round(asyn, 3)

        # If you want to save the data:
        # cleaned_data = pd.DataFrame(list(zip(iti, asyn, ioi)))
        # cleaned_data.to_csv("cleaned_data.csv", header=None, index=None)

        # Variables for behavioural data
        median_abs_asyn = np.median(np.abs(asyn)).round(3)
        min_abs_asyn = np.min(np.abs(asyn))
        max_abs_asyn = np.max(np.abs(asyn))
        mean_asyn = np.mean(asyn).round(3)
        # N.B.: SD is different from MATLAB - divides by N instead of N-1
        sd_asyn = np.std(asyn).round(3)
        mean_abs_asyn = np.mean(np.abs(asyn)).round(3)
        sd_abs_asyn = np.std(np.abs(asyn)).round(3)
        min_asyn = np.min(asyn)
        max_asyn = np.max(asyn)
        median_iti = np.median(iti)
        mean_iti = np.mean(iti).round(3)
        sd_iti = np.std(iti).round(3)
        min_iti = np.min(iti)
        max_iti = np.max(iti)
        median_ioi = np.median(ioi)
        mean_ioi = np.mean(ioi).round(3)
        sd_ioi = np.std(ioi).round(3)
        min_ioi = np.min(ioi)
        max_ioi = np.max(ioi)
        cv_iti = (sd_iti / mean_iti).round(3)
        cv_asyn = (sd_asyn / mean_iti).round(3)
        n_events = len(ioi)

        # These lines estimate the parameters for each model selected,
        # and sets the parameters not generated by a particular model to NaN.
        for model in models:
            if model == "joint_model_beta":
                gamma, mE, beta, sT, sM, LL = joint_model_beta(
                    ioi, iti, asyn, 0, 1.1
                )
                alpha = np.nan
            elif model == "adaptation":
                alpha, beta, sM, sT, LL = adaptation_model(iti, asyn)
                mE = np.nan
                gamma = np.nan

            # Saves the results in a dataframe.
            results.loc[len(results.index)] = [
                file,
                leader,
                follower,
                model,
                alpha,
                beta,
                mE,
                gamma,
                sT,
                sM,
                LL,
                median_abs_asyn,
                min_abs_asyn,
                max_abs_asyn,
                mean_asyn,
                sd_asyn,
                mean_abs_asyn,
                sd_abs_asyn,
                min_asyn,
                max_asyn,
                median_iti,
                mean_iti,
                sd_iti,
                min_iti,
                max_iti,
                median_ioi,
                mean_ioi,
                sd_ioi,
                min_ioi,
                max_ioi,
                cv_iti,
                cv_asyn,
                n_events,
                n_interpolations,
            ]

# Saves all of the results to this file.
results.to_csv("example_results.csv", index=None)
