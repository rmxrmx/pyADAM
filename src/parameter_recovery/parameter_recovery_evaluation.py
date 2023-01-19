"""
Script for analysing how well the model predicts parameters.

Usage:

Write in the location of the result file.
Set the threshold of SD of asynchrony (runs with average SD above that will not be used).
Choose which models to run (available: jointmodelbeta, adaptation).
Choose whether the plots should be saved.

N.B.: results file must have the following columns:
model, sd_async, actual_alpha, actual_beta, actual_delta, actual_phi, alpha, beta, delta, phi
"""

import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit


results_to_analyse = pd.read_csv("new_interr_results.csv")
SD_THRESHOLD = 300
models = ["jointmodelbeta", "adaptation"]
SAVE_PLOTS = True
PLOT_LOCATION = "recovery_plots"
# TODO: create plot directory if it doesn't exist

for model in models:

    # get data only for this model
    data = results_to_analyse[results_to_analyse["model"] == model]

    data = data[data["sd_async"] <= SD_THRESHOLD]

    # plot histogram with 1000 bins
    fig, ax = plt.subplots()
    ax.hist(data["sd_async"], bins=1000)
    ax.set_xlabel("SD of asynchronies")
    ax.set_ylabel("n")
    ax.set_title("Histogram of SDs of asynchronies")

    grouped = data.groupby(["actual_alpha", "actual_beta", "actual_delta", "actual_phi"])

    # choose which parameters to plot
    if model == "adaptation":
        relevant_parameters = ["alpha", "beta"]
    elif model == "jointmodelbeta":
        relevant_parameters = ["beta", "delta", "phi"]

    fig, ax = plt.subplots(len(relevant_parameters), 1)
    ax[0].set_title(f"Parameter estimations for the {model} model")

    # make a plot for each parameter
    for index, parameter in enumerate(relevant_parameters):

        x = grouped[f"{parameter}"].mean().to_numpy()
        y = grouped[f"actual_{parameter}"].mean().to_numpy()

        # formatting plots
        ax[index].set_axisbelow(True)
        ax[index].yaxis.grid()
        hb = ax[index].hexbin(x, y, gridsize=(40,5), cmap="viridis", mincnt=1)
        ax[index].set_ylim(-0.1, 1.2)
        ax[index].set_xlabel(f"predicted {parameter}")
        ax[index].set_ylabel(f"actual {parameter}")
        fig.colorbar(hb, ax = ax[index])
        if parameter == "alpha":
            ax[index].set_yticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
        elif parameter == "beta":
            ax[index].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif parameter == "phi":
            ax[index].set_yticks([0.1, 0.5, 0.9])
        else:
            ax[index].set_yticks([0, 0.5, 1.0])

        # add a trendline
        b, m = polyfit(x, y, 1)
        ax[index].plot(x, b + m * x, '-')

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_LOCATION}/{model}.png")
    plt.show()
