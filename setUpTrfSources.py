# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: neurotools
# ---

# %%
import mne
import numpy as np
import eelbrain as eb
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
from nilearn import plotting

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *
from computeTrfsGeneralized import computeTrfs, computeSources

matplotlib.use("QtAgg")
plt.ion()

# %load_ext autoreload
# %autoreload 2

# %%


subjects = [
    "R3045",
    "R3089",
    "R3093",
    "R3095",
    "R3151",
    "R2774",
    "R3152",
    "R2877",
    "R3157",
    "R2783",
]
useAvgBrains = [False, False, False, False, True, False, False, True, False, False]
badChanLists = [
    None,
    ["P7", "CP6", "C4", "T7", "CP5", "P3", "P4", "O2", "Oz", "PO4"],
    ["Oz", "P8", "CP6", "Fp2"],
    ["P7", "T8", "O2", "PO4"],
    ["C3", "FC5", "P4"],
    [
        "Fp1",
        "AF3",
        "F7",
        "F3",
        "Fz",
        "F4",
        "FC6",
        "C3",
        "CP5",
        "Pz",
        "CP6",
        "P8",
        "FC1",
        "FC5",
        "T7",
        "AF4",
    ],
    ["T7", "C3", "P7", "Pz", "O1", "P8", "CP6"],
    ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"],
    None,
    None,
]

# parentDir = "/Users/karl/map/"
parentDir = "/Volumes/Seagate/map/"

# subjects = ["R2877"]
# useAvgBrains = [True]
# badChanLists = [None]


typeOfRegressors = ["mix", "mix", "target", "distractor", "target", "distractor"]
filenameSuffixes = [
    "_quiet_male",
    "_quiet_female",
    "_male",
    "_male",
    "_female",
    "_female",
]

nameOfRegressors = ["_ANmodel_correctedLevels", "~gammatone-1", "~gammatone-on-1"]
bandpassFreqsList = [[20, 1000], [2, None], [2, None]]


for iSubject, subject in enumerate(subjects):
    for iType, typeOfRegressor in enumerate(typeOfRegressors):
        for iName, nameOfRegressor in enumerate(nameOfRegressors):

            print("\n")
            print(
                [
                    parentDir,
                    subject,
                    useAvgBrains[iSubject],
                    badChanLists[iSubject],
                    typeOfRegressor,
                    nameOfRegressor,
                    bandpassFreqsList[iName],
                    filenameSuffixes[iType],
                ]
            )
            print("\n")
            computeSources(
                parentDir,
                subject,
                useAvgBrains[iSubject],
                badChanLists[iSubject],
                typeOfRegressor,
                nameOfRegressor,
                bandpassFreqsList[iName],
                filenameSuffixes[iType],
            )
