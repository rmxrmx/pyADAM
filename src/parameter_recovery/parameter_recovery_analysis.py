"""
A script for analysing SMS data using ADAM models.
"""
import glob
import numpy as np
import pandas as pd
from src.adam_modules.joint_model_beta import joint_model_beta
from src.adam_modules.adaptation_model import adaptation_model

# A results dataframe for storing behavioural data and parameter estimations
results = pd.DataFrame(
    columns=[
        "actual_alpha",
        "actual_beta",
        "actual_delta",
        "actual_phi",
        "model",
        "alpha",
        "beta",
        "delta",
        "phi",
        "sd_async",
    ]
)

# This variable should contain all of the files that are to be analysed.
files = glob.glob("parameter_recovery_data/*.csv")

for file in files:
    data = pd.read_csv(file, header=None)

    # assume that ITI was follower, IOI was leader
    groups = [[0, 1]]

    for grouping in groups:
        follower, leader = grouping

        # Data is in format of ITI, asyn, IOI.
        # Skip the first few datapoints as the participants
        # might not be synchronized on startup.
        iti = data.iloc[:, 0].tolist()[4:]
        ioi = data.iloc[:, 2].tolist()[4:]
        asyn = data.iloc[:, 1].tolist()[4:]

        # Behavioural data
        # N.B.: SD is different from MATLAB - divides by N instead of N-1
        sd_asyn = np.std(asyn).round(3)

        # Split by _ to get the actual values of the run.
        # Note: this part of the script assumes the datafiles are named
        # parameter_recovery_data/interrogator_results_{alpha_value}_{beta_value}_{delta_value}_{phi_value}_run{number}_{model}.csv
        # Your data can be named something else, but the lines below should be changed accordingly.
        params = file.split("_")
        actual_alpha = params[4]
        actual_beta = params[5]
        actual_delta = params[6]
        actual_phi = params[7]
        model = params[9][:-4]

        # calculate the parameters
        if model == "jointmodelbeta":
            gamma, mE, beta, sT, sM, LL = joint_model_beta(
                ioi, iti, asyn, 0, 1.1
            )
            alpha = np.nan
        elif model == "adaptation":
            alpha, beta, sM, sT, LL = adaptation_model(iti, asyn)
            mE = np.nan
            gamma = np.nan

        # bind beta between 0 and 1
        if beta > 1:
            beta = 1
        elif beta < 0:
            beta = 0

        results.loc[len(results.index)] = [
            actual_alpha,
            actual_beta,
            actual_delta,
            actual_phi,
            model,
            alpha,
            beta,
            mE,
            gamma,
            sd_asyn,
        ]

results.to_csv("parameter_recovery_results.csv", index=None)
