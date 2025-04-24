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
import joblib
from nilearn import plotting

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *

from computeTrfsMEEG import computeTrfs, computeSources

# from computeTrfsEEG import computeTrfs, computeSources

matplotlib.use("QtAgg")
plt.ion()

# %load_ext autoreload
# %autoreload 2

# %%

# %%
# parentDir = "/Users/karl/map/"
parentDir = "/Volumes/Seagate/map/"

# subjects = ["R3172"]
# useAvgBrains = [True]
# badChanLists = [None]


doParallel = True
n_jobs = 16
backend = "loky"
verbose = 49


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
    "R3170",
    "R3172",
    "R3193",
    "R3184",
    "R3214",
]

subset = [i for i in range(len(subjects))]
# subset = [6,7,8,9,10,11,12,13,14]
subjects = [subjects[i] for i in subset]

useAvgBrains = [
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
]
useAvgBrains = [useAvgBrains[i] for i in subset]

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
    None,
    None,
    None,
    None,
    None,
]
badChanLists = [badChanLists[i] for i in subset]


typeOfRegressors = ["mix", "target", "target", "distractor", "distractor"]
filenameSuffixes = [
    "_quiet",
    "_easy",
    "_hard",
    "_easy",
    "_hard",
]


# nameOfRegressors = ["_ANmodel_maxFs"]

nameOfRegressors = [
    "~gammatone-1",
    "~gammatone-on-1",
    "~wordOnsets_gaussian15msSD",
    "~phoneOnsets_gaussian15msSD",
    "~phsurp",
    "~cohtent",
    "~wordprob",
    "~wordsurp",
]

bandpassFreqsList = [
    [2, None],
    [2, None],
    [2, None],
    [2, None],
    [2, None],
    [2, None],
    [2, None],
    [2, None],
]
# bandpassFreqsList = [[20, 1000]]


# newestSubjectsToDo = 2  # Use this to do only the N most recent subjects, meaning the order of the subject list variable above
# newestSubjectsToDo = len(
#     subjects
# )  # Use this to do all subjects, so we can keep the first line of the loop the way it is

# for iSubject, subject in enumerate(subjects):
# for iSubject, subject in enumerate(
#     subjects[-newestSubjectsToDo:], start=len(subjects) - newestSubjectsToDo
# ):


def doOneSubject(
    iSubject,
    subjects,
    typeOfRegressors,
    nameOfRegressors,
    parentDir,
    useAvgBrains,
    badChanLists,
    bandpassFreqsList,
    filenameSuffixes,
):

    subject = subjects[iSubject]

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


# %%
if doParallel:

    joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        joblib.delayed(doOneSubject)(
            iSubject,
            subjects,
            typeOfRegressors,
            nameOfRegressors,
            parentDir,
            useAvgBrains,
            badChanLists,
            bandpassFreqsList,
            filenameSuffixes,
        )
        for iSubject in range(len(subjects))
    )

else:

    for iSubject in range(len(subjects)):
        doOneSubject(
            iSubject,
            subjects,
            typeOfRegressors,
            nameOfRegressors,
            parentDir,
            useAvgBrains,
            badChanLists,
            bandpassFreqsList,
            filenameSuffixes,
        )
