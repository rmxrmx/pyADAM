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
        "filename",
        "run",
        "actual_alpha",
        "actual_beta",
        "actual_delta",
        "actual_phi",
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

        # Split by _ to get the actual values of the run.
        # Note: this part of the script assumes the datafiles are named
        # parameter_recovery_data/interrogator_results_{alpha_value}_{beta_value}_{delta_value}_{phi_value}_run{number}_{model}.csv
        # Your data can be named something else, but the lines below should be changed accordingly.
        params = file.split("_")
        actual_alpha = params[4]
        actual_beta = params[5]
        actual_delta = params[6]
        actual_phi = params[7]
        run_no = params[8][3:]
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
            file,
            run_no,
            actual_alpha,
            actual_beta,
            actual_delta,
            actual_phi,
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
        ]

results.to_csv("parameter_recovery_results.csv", index=None)
