"""
A script for analysing SMS data using ADAM models.
"""
import glob
import numpy as np
import pandas as pd
from src.joint_model_beta import joint_model_beta
from src.adaptation_model import adaptation_model
from src.qa_data import convert_to_intervals, interpolate_onsets

# A results dataframe for storing behavioural data and parameter estimations
results = pd.DataFrame(
    columns=[
        "filename",
        "run",
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
        "n_interpolated"
    ]
)

# This variable should contain all of the files that are to be analysed.
files = glob.glob("*session1*.csv")
for file in files:

    # Read in an Italian-style CSV
    data = pd.read_csv(file, decimal=",", delimiter=";")

    # change commas to dots, cast them as floats and convert seconds into milliseconds
    data["RT_1 s"] = (
        data["RT_1 s"].str.replace(",", ".").replace("None", None).astype(float) * 1000
    )
    # TODO: some issues with this line
    # data[" key_tone_onset"] = data[" key_tone_onset"].str.replace(',', '.').replace("None", None).astype(float) * 1000
    data["RT_2 s"] = (
        data["RT_2 s"].str.replace(",", ".").replace("None", None).astype(float) * 1000
    )

    # Group the data by the run
    runs = [x for _, x in data.groupby(data["run"])]

    for data in runs:
        # This section interpolates the onsets and calculates the ITI, IOI and async from them
        onsets1, n_interpolated1 = interpolate_onsets(data["RT_1 s"].tolist())
        onsets2, n_interpolated2 = interpolate_onsets(data["RT_2 s"].tolist())

        iti, ioi, asyn = convert_to_intervals(onsets1, onsets2)
        n_interpolated = n_interpolated1 + n_interpolated2

        iti = np.round(iti, 3)
        ioi = np.round(ioi, 3)
        asyn = np.round(asyn, 3)

        # If you want to save the data:
        # quality_data = pd.DataFrame(list(zip(iti, asyn, ioi)))
        # quality_data.to_csv("cleaned_data.csv", header=None, index=None)

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

        models = ["joint_model_beta", "adaptation"]

        for model in models:
            if model == "joint_model_beta":
                gamma, mE, beta, sT, sM, LL = joint_model_beta(ioi, iti, asyn, 0, 1.1)
                alpha = np.nan
            elif model == "adaptation":
                alpha, beta, sM, sT, LL = adaptation_model(iti, asyn)
                mE = np.nan
                gamma = np.nan

            results.loc[len(results.index)] = [
                file,
                data.iloc[0]["run"],
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
                n_interpolated
            ]

results.to_csv("results.csv", index=None)
