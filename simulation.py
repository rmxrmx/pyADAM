import numpy as np
from src.joint_model_beta import joint_model_beta
from src.adaptation_model import adaptation_model
import pandas as pd
import glob

results = pd.DataFrame(columns=["filename", "alpha", "beta", "delta", "phi", "sT", "sM", "LLE", "median_abs_async", "min_abs_async", "max_abs_async",
                                "mean_async", "sd_async", "min_asyn", "max_asyn", "median_ITI", "sd_ITI", "min_ITI", "max_ITI", "median_IOI", "sd_IOI",
                                "min_IOI", "max_IOI", "n_events"])

files = glob.glob("*cleaned*.csv")
for file in files:

    data = pd.read_csv(file, header=None)

    # remove long IOIs - to discuss
    data = data[data[2] <= 6000]

    iti = data[0].tolist()
    ioi = data[2].tolist()
    asyn = data[1].tolist()

    median_abs_asyn = np.median(np.abs(asyn))
    min_abs_asyn = np.min(np.abs(asyn))
    max_abs_asyn = np.max(np.abs(asyn))
    mean_asyn = np.mean(asyn)
    sd_asyn = np.std(asyn)
    min_asyn = np.min(asyn)
    max_asyn = np.max(asyn)
    median_iti = np.median(iti)
    sd_iti = np.std(iti)
    min_iti = np.min(iti)
    max_iti = np.max(iti)
    median_ioi = np.median(ioi)
    # different from MATLAB - divides by N instead of N-1
    sd_ioi = np.std(ioi)
    min_ioi = np.min(ioi)
    max_ioi = np.max(ioi)
    n_events = len(ioi)


    # TODO: number of iterations and k-ratio should be passed as parameters

    gammaE, mE, betaE, stE, smE, LL = joint_model_beta(ioi, iti, asyn, -2, 2)
    results.loc[len(results.index)] = [file, np.nan, betaE, mE, gammaE, stE, smE, LL, median_abs_asyn, min_abs_asyn, max_abs_asyn, mean_asyn, sd_asyn, min_asyn,
                                       max_asyn, median_iti, sd_iti, min_iti, max_iti, median_ioi, sd_ioi, min_ioi, max_ioi, n_events]
    
    alpha, beta, sM, sT, LL = adaptation_model(iti, asyn)
    results.loc[len(results.index)] = [file, alpha, beta, np.nan, np.nan, sT, sM, LL, median_abs_asyn, min_abs_asyn, max_abs_asyn, mean_asyn, sd_asyn, min_asyn,
                                        max_asyn, median_iti, sd_iti, min_iti, max_iti, median_ioi, sd_ioi, min_ioi, max_ioi, n_events]


results.to_csv("results_joint_beta.csv", index=None)