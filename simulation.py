import glob
import numpy as np
import pandas as pd
from src.joint_model_beta import joint_model_beta
from src.adaptation_model import adaptation_model
from src.qa_data import convert_to_intervals, interpolate_onsets

results = pd.DataFrame(
    columns=[
        "filename",
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

files = glob.glob("*session1*.csv")
for file in files:

    data = pd.read_csv(file, decimal=",", delimiter=";")
    data = data[np.isnan(data["errCondition"])]

    # TODO: the likelihood comes out to 0 - need to figure out the issue and fix it
    # change commas to dots, cast them as floats and convert seconds into milliseconds
    data["RT_1 s"] = (
        data["RT_1 s"].str.replace(",", ".").replace("None", None).astype(float) * 1000
    )
    # TODO: some issues with this line
    # data[" key_tone_onset"] = data[" key_tone_onset"].str.replace(',', '.').replace("None", None).astype(float) * 1000
    data["RT_2 s"] = (
        data["RT_2 s"].str.replace(",", ".").replace("None", None).astype(float) * 1000
    )

    data1 = data[data["run"] == 1]
    data1 = data.reset_index()
    data2 = data[data["run"] != 1]
    data2.reset_index()

    onsets1 = interpolate_onsets(data2["RT_1 s"].tolist())
    onsets2 = interpolate_onsets(data2["RT_2 s"].tolist())

    iti, ioi, asyn = convert_to_intervals(onsets1, onsets2)

    iti = np.round(iti, 3)
    ioi = np.round(ioi, 3)
    asyn = np.round(asyn, 3)

    quality_data = pd.DataFrame(list(zip(iti, asyn, ioi)))
    quality_data.to_csv("cleaned_data.csv", header=None, index=None)

    # remove long IOIs - this should likely be moved to QA
    # data = data[data[2] <= 6000]

    # iti = data[0].tolist()
    # ioi = data[2].tolist()
    # asyn = data[1].tolist()

    median_abs_asyn = np.median(np.abs(asyn))
    min_abs_asyn = np.min(np.abs(asyn))
    max_abs_asyn = np.max(np.abs(asyn))
    mean_asyn = np.mean(asyn)
    sd_asyn = np.std(asyn)
    mean_abs_asyn = np.mean(np.abs(asyn))
    sd_abs_asyn = np.std(np.abs(asyn))
    min_asyn = np.min(asyn)
    max_asyn = np.max(asyn)
    median_iti = np.median(iti)
    mean_iti = np.mean(iti)
    sd_iti = np.std(iti)
    min_iti = np.min(iti)
    max_iti = np.max(iti)
    median_ioi = np.median(ioi)
    mean_ioi = np.mean(ioi)
    # different from MATLAB - divides by N instead of N-1
    sd_ioi = np.std(ioi)
    min_ioi = np.min(ioi)
    max_ioi = np.max(ioi)
    cv_iti = sd_iti / mean_iti
    cv_asyn = sd_asyn / mean_iti
    n_events = len(ioi)

    gammaE, mE, betaE, stE, smE, LL = joint_model_beta(ioi, iti, asyn, 0, 1.1)
    results.loc[len(results.index)] = [
        file,
        np.nan,
        betaE,
        mE,
        gammaE,
        stE,
        smE,
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

    alpha, beta, sM, sT, LL = adaptation_model(iti, asyn)
    results.loc[len(results.index)] = [
        file,
        alpha,
        beta,
        np.nan,
        np.nan,
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

# TODO: results need to either be divided by model or specify model in a column
results.to_csv("results.csv", index=None)
