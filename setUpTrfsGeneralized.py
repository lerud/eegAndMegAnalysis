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
import mat73
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pickle
import glob
import time
import gc

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *

# from computeTrfsEEG import computeTrfs
from computeTrfsMEEG import computeTrfs

matplotlib.use("QtAgg")
plt.ion()

# %load_ext autoreload
# %autoreload 2

# %%
# parentDir = "/Users/karl/map/"
parentDir = "/Volumes/Seagate/map/"

doParallel = True
n_jobs = 4
backend = "loky"
verbose = 49

lambdas = np.logspace(-1, 9, 15)

conditionNames = dict(
    A=[
        "FemaleBrit_over_FemaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Easy",
        "MaleBrit___Quiet",
        "MaleBrit___Quiet",
        "FemaleAmer_over_FemaleBrit_Hard",
        "FemaleAmer_over_FemaleBrit_Hard",
        "MaleAmer___over_MaleBrit_Easy",
        "MaleAmer___over_MaleBrit_Easy",
        "FemaleAmer_Quiet",
        "FemaleAmer_Quiet",
        "MaleAmer___over_FemaleAmer_Easy",
        "MaleAmer___over_FemaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Hard",
        "FemaleBrit_over_FemaleAmer_Hard",
        "MaleBrit___over_MaleAmer_Easy",
        "MaleBrit___over_MaleAmer_Easy",
        "FemaleAmer_over_MaleAmer_Hard",
        "FemaleAmer_over_MaleAmer_Hard",
        "MaleAmer___Quiet",
        "MaleAmer___Quiet",
        "FemaleAmer_over_MaleAmer_Easy",
        "FemaleAmer_over_MaleAmer_Easy",
        "MaleBrit___over_MaleAmer_Hard",
        "MaleBrit___over_MaleAmer_Hard",
        "FemaleBrit_Quiet",
        "FemaleBrit_Quiet",
        "MaleAmer___over_MaleBrit_Hard",
        "MaleAmer___over_MaleBrit_Hard",
        "FemaleAmer_over_FemaleBrit_Easy",
        "FemaleAmer_over_FemaleBrit_Easy",
        "MaleAmer___over_FemaleAmer_Hard",
        "MaleAmer___over_FemaleAmer_Hard",
    ],
    B=[
        "MaleBrit___over_MaleAmer_Easy",
        "MaleBrit___over_MaleAmer_Easy",
        "FemaleBrit_Quiet",
        "FemaleBrit_Quiet",
        "MaleAmer___over_FemaleAmer_Hard",
        "MaleAmer___over_FemaleAmer_Hard",
        "FemaleAmer_over_FemaleBrit_Easy",
        "FemaleAmer_over_FemaleBrit_Easy",
        "MaleBrit___Quiet",
        "MaleBrit___Quiet",
        "FemaleBrit_over_FemaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Easy",
        "MaleAmer___over_MaleBrit_Hard",
        "MaleAmer___over_MaleBrit_Hard",
        "FemaleAmer_Quiet",
        "FemaleAmer_Quiet",
        "MaleBrit___over_MaleAmer_Hard",
        "MaleBrit___over_MaleAmer_Hard",
        "FemaleAmer_over_MaleAmer_Easy",
        "FemaleAmer_over_MaleAmer_Easy",
        "MaleAmer___Quiet",
        "MaleAmer___Quiet",
        "FemaleAmer_over_FemaleBrit_Hard",
        "FemaleAmer_over_FemaleBrit_Hard",
        "MaleAmer___over_FemaleAmer_Easy",
        "MaleAmer___over_FemaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Hard",
        "FemaleBrit_over_FemaleAmer_Hard",
        "MaleAmer___over_MaleBrit_Easy",
        "MaleAmer___over_MaleBrit_Easy",
        "FemaleAmer_over_MaleAmer_Hard",
        "FemaleAmer_over_MaleAmer_Hard",
    ],
    C=[
        "FemaleAmer_over_MaleAmer_Easy",
        "FemaleAmer_over_MaleAmer_Easy",
        "MaleAmer___Quiet",
        "MaleAmer___Quiet",
        "FemaleAmer_over_FemaleBrit_Hard",
        "FemaleAmer_over_FemaleBrit_Hard",
        "MaleAmer___over_FemaleAmer_Easy",
        "MaleAmer___over_FemaleAmer_Easy",
        "FemaleAmer_Quiet",
        "FemaleAmer_Quiet",
        "MaleAmer___over_MaleBrit_Easy",
        "MaleAmer___over_MaleBrit_Easy",
        "FemaleBrit_over_FemaleAmer_Hard",
        "FemaleBrit_over_FemaleAmer_Hard",
        "MaleBrit___over_MaleAmer_Easy",
        "MaleBrit___over_MaleAmer_Easy",
        "FemaleAmer_over_MaleAmer_Hard",
        "FemaleAmer_over_MaleAmer_Hard",
        "MaleBrit___Quiet",
        "MaleBrit___Quiet",
        "FemaleAmer_over_FemaleBrit_Easy",
        "FemaleAmer_over_FemaleBrit_Easy",
        "MaleAmer___over_MaleBrit_Hard",
        "MaleAmer___over_MaleBrit_Hard",
        "FemaleBrit_over_FemaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Easy",
        "MaleBrit___over_MaleAmer_Hard",
        "MaleBrit___over_MaleAmer_Hard",
        "FemaleBrit_Quiet",
        "FemaleBrit_Quiet",
        "MaleAmer___over_FemaleAmer_Hard",
        "MaleAmer___over_FemaleAmer_Hard",
    ],
    D=[
        "MaleAmer___over_FemaleAmer_Easy",
        "MaleAmer___over_FemaleAmer_Easy",
        "FemaleBrit_Quiet",
        "FemaleBrit_Quiet",
        "MaleBrit___over_MaleAmer_Hard",
        "MaleBrit___over_MaleAmer_Hard",
        "FemaleAmer_over_MaleAmer_Easy",
        "FemaleAmer_over_MaleAmer_Easy",
        "MaleAmer___Quiet",
        "MaleAmer___Quiet",
        "FemaleAmer_over_FemaleBrit_Hard",
        "FemaleAmer_over_FemaleBrit_Hard",
        "MaleBrit___Quiet",
        "MaleBrit___Quiet",
        "FemaleAmer_over_FemaleBrit_Easy",
        "FemaleAmer_over_FemaleBrit_Easy",
        "MaleAmer___over_MaleBrit_Hard",
        "MaleAmer___over_MaleBrit_Hard",
        "FemaleBrit_over_FemaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Easy",
        "MaleAmer___over_FemaleAmer_Hard",
        "MaleAmer___over_FemaleAmer_Hard",
        "FemaleAmer_Quiet",
        "FemaleAmer_Quiet",
        "MaleBrit___over_MaleAmer_Easy",
        "MaleBrit___over_MaleAmer_Easy",
        "FemaleBrit_over_FemaleAmer_Hard",
        "FemaleBrit_over_FemaleAmer_Hard",
        "MaleAmer___over_MaleBrit_Easy",
        "MaleAmer___over_MaleBrit_Easy",
        "FemaleAmer_over_MaleAmer_Hard",
        "FemaleAmer_over_MaleAmer_Hard",
    ],
)

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
# subset = np.arange(1, len(subjects))
subjects = [subjects[i] for i in subset]

presStopCorrections = [
    0,
    0,
    0,
    0,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]
presStopCorrections = [presStopCorrections[i] for i in subset]

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

conditions = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C"]
conditions = [conditions[i] for i in subset]


# typeOfRegressors = ["mix", "target", "target", "distractor", "distractor"]
# typeOfRegressors = ["mix", "mix", "target", "target", "distractor", "distractor"]
# typeOfRegressors = ["target", "target", "target", "distractor", "distractor", "distractor", "mix", "mix", "mix", "mix", "mix"]

# This one is a concatenation of the three groups above
typeOfRegressors = [
    "mix",
    "target",
    "target",
    "distractor",
    "distractor",
    "mix",
    "mix",
    "target",
    "target",
    "distractor",
    "distractor",
    "target",
    "target",
    "target",
    "distractor",
    "distractor",
    "distractor",
    "mix",
    "mix",
    "mix",
    "mix",
    "mix",
]


# regressorDirs = [
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
# ]
# regressorDirs = [
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
# ]
# regressorDirs = [
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/targets/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
#     parentDir + "stimAndPredictors/distractors/predictors/",
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/mixes/predictors/",
#     parentDir + "stimAndPredictors/mixes/predictors/",
# ]

# This one is a concatenation of the three groups above
regressorDirs = [
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/targets/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/distractors/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
    parentDir + "stimAndPredictors/mixes/predictors/",
]


# filenameSuffixes = ["_quiet", "_easy", "_hard", "_easy", "_hard"]
# filenameSuffixes = ["_quiet_male", "_quiet_female", "_male", "_female", "_male", "_female"]
# filenameSuffixes = ["_4dB", "_0dB", "_-4dB", "_4dB", "_0dB", "_-4dB", "_4dB", "_0dB", "_-4dB", "_easy", "_hard"]

# This one is a concatenation of the three groups above
filenameSuffixes = [
    "_quiet",
    "_easy",
    "_hard",
    "_easy",
    "_hard",
    "_quiet_male",
    "_quiet_female",
    "_male",
    "_female",
    "_male",
    "_female",
    "_4dB",
    "_0dB",
    "_-4dB",
    "_4dB",
    "_0dB",
    "_-4dB",
    "_4dB",
    "_0dB",
    "_-4dB",
    "_easy",
    "_hard",
]

# nameOfRegressors = ["_ANmodel_maxFs"]
# nameOfRegressors = ["_ANmodel_correctedLevels", "~gammatone-1", "~gammatone-on-1"]
# nameOfRegressors = ["~wordOnsets", "~phoneOnsets"]
# nameOfRegressors = ["~wordOnsets_gaussian", "~phoneOnsets_gaussian"]
# nameOfRegressors = ["~wordOnsets_gaussian20SD", "~phoneOnsets_gaussian20SD"]
# nameOfRegressors = ["~wordOnsets_gaussian40SD", "~phoneOnsets_gaussian40SD"]

# nameOfRegressors = [
#     "_ANmodel_maxFs",
#     "~gammatone-1",
#     "~gammatone-on-1",
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordprob",
#     "~wordsurp",
# ]

# This needs to be a list of lists now, so to run multiple regressors at once, make it a list with one element, 
# which is a list of strings:
# nameOfRegressors = [[
#     "~gammatone-1",
#     "~gammatone-on-1",
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordprob",
#     "~wordsurp",
# ]]

# This needs to be a list of lists now, so to run multiple regressors at once, make it a list with one element, 
# which is a list of strings:
nameOfRegressors = [[
    "~gammatone-1",
    "~gammatone-on-1",
    "~wordOnsets_gaussian15msSD",
    "~phoneOnsets_gaussian15msSD",
    "~phsurp",
    "~cohtent",
    "~wordsurp",
]]

# Acoustic only for acoustic rf
# nameOfRegressors = [[
#     "~gammatone-1",
#     "~gammatone-on-1",
# ]]


# Linguistic only 
# nameOfRegressors = [[
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordprob",
#     "~wordsurp",
# ]]


# Taking word probability out of the linguistic ones for now because we're not using it or interpreting it
# nameOfRegressors = [[
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordsurp",
# ]]

# Alternatively, to run each regressor separately, make it a list of lists again, but where each element is a list 
# itself with only one string:
# nameOfRegressors = [
#     ["~gammatone-1"],
#     ["~gammatone-on-1"],
#     ["~wordOnsets_gaussian15msSD"],
#     ["~phoneOnsets_gaussian15msSD"],
#     ["~phsurp"],
#     ["~cohtent"],
#     ["~wordprob"],
#     ["~wordsurp"],
# ]

# nameOfRegressors = [[
#     "~gammatone-1",
# ]]

# filenamePrefix = "concurRecField"
# filenamePrefix = "acousticConcurRecField"
# filenamePrefix = "acousticRecFieldAllFolds"
# filenamePrefix = "linguisticConcurRecField"
# filenamePrefix = "linguisticRecFieldAllFolds"
filenamePrefix = "combinedRecFieldAllFolds"

# newestSubjectsToDo = 1  # Use this to do only the N most recent subjects, meaning the order of the subject list variable above
# newestSubjectsToDo = len(
#     subjects
# )  # Use this to do all subjects, so we can keep the first line of the loop the way it is

# %%
# for iSubject, subject in enumerate(
#     subjects[-newestSubjectsToDo:], start=len(subjects) - newestSubjectsToDo
# ):

def doOneSubject(iSubject, subjects, typeOfRegressors, nameOfRegressors, regressorDirs, conditions, filenamePrefix, filenameSuffixes, conditionNames, parentDir, presStopCorrections, badChanLists, lambdaInd, thisLambda):

    subject = subjects[iSubject]
    
    for iType, typeOfRegressor in enumerate(typeOfRegressors):
        for nameOfRegressor in nameOfRegressors:

            regressorDir = regressorDirs[iType]
            condition = conditions[iSubject]
            filenameSuffix = filenameSuffixes[iType]

            trialsToAnalyze = []

            for iTrial in range(32):

                conditionName = conditionNames[condition][iTrial]

                if len(conditionName) > 16:

                    # t is quiet, y is easy, d is hard
                    if filenameSuffix == "_easy" and conditionName[-1] == "y":
                        trialsToAnalyze.append(iTrial)
                    if filenameSuffix == "_hard" and conditionName[-1] == "d":
                        trialsToAnalyze.append(iTrial)

                    # M is male, F is female
                    if (
                        filenameSuffix == "_male"
                        and typeOfRegressor == "target"
                        and conditionName[0] == "M"
                    ):
                        trialsToAnalyze.append(iTrial)
                    if (
                        filenameSuffix == "_male"
                        and typeOfRegressor == "distractor"
                        and conditionName[16] == "M"
                    ):
                        trialsToAnalyze.append(iTrial)
                    if (
                        filenameSuffix == "_female"
                        and typeOfRegressor == "target"
                        and conditionName[0] == "F"
                    ):
                        trialsToAnalyze.append(iTrial)
                    if (
                        filenameSuffix == "_female"
                        and typeOfRegressor == "distractor"
                        and conditionName[16] == "F"
                    ):
                        trialsToAnalyze.append(iTrial)

                    # M is male, F is female, y is easy, d is hard
                    if filenameSuffix == "_4dB":
                        if (
                            conditionName[0] == "M"
                            and conditionName[16] == "M"
                            and conditionName[-1] == "y"
                        ):
                            trialsToAnalyze.append(iTrial)
                        elif (
                            conditionName[0] == "F"
                            and conditionName[16] == "F"
                            and conditionName[-1] == "y"
                        ):
                            trialsToAnalyze.append(iTrial)
                    if filenameSuffix == "_0dB":
                        if (
                            conditionName[0] == "M"
                            and conditionName[16] == "M"
                            and conditionName[-1] == "d"
                        ):
                            trialsToAnalyze.append(iTrial)
                        elif (
                            conditionName[0] == "F"
                            and conditionName[16] == "F"
                            and conditionName[-1] == "d"
                        ):
                            trialsToAnalyze.append(iTrial)
                        elif (
                            conditionName[0] == "M"
                            and conditionName[16] == "F"
                            and conditionName[-1] == "y"
                        ):
                            trialsToAnalyze.append(iTrial)
                        elif (
                            conditionName[0] == "F"
                            and conditionName[16] == "M"
                            and conditionName[-1] == "y"
                        ):
                            trialsToAnalyze.append(iTrial)
                    if filenameSuffix == "_-4dB":
                        if (
                            conditionName[0] == "M"
                            and conditionName[16] == "F"
                            and conditionName[-1] == "d"
                        ):
                            trialsToAnalyze.append(iTrial)
                        elif (
                            conditionName[0] == "F"
                            and conditionName[16] == "M"
                            and conditionName[-1] == "d"
                        ):
                            trialsToAnalyze.append(iTrial)

                elif len(conditionName) == 16:

                    # t is quiet, y is easy, d is hard
                    if filenameSuffix == "_quiet" and conditionName[-1] == "t":
                        trialsToAnalyze.append(iTrial)

                    # M is male, F is female
                    if filenameSuffix == "_quiet_male" and conditionName[0] == "M":
                        trialsToAnalyze.append(iTrial)
                    if filenameSuffix == "_quiet_female" and conditionName[0] == "F":
                        trialsToAnalyze.append(iTrial)

                # if filenameSuffix == "_easy" and conditionName[-1] == "y":
                #     trialsToAnalyze.append(iTrial)
                # if filenameSuffix == "_hard" and conditionName[-1] == "d":
                #     trialsToAnalyze.append(iTrial)

            trialsToAnalyze = np.array(trialsToAnalyze)

            print("\n")
            print(
                [
                    parentDir,
                    subject,
                    presStopCorrections[iSubject],
                    badChanLists[iSubject],
                    condition,
                    typeOfRegressor,
                    nameOfRegressor,
                    regressorDir,
                    trialsToAnalyze,
                    filenamePrefix,
                    filenameSuffix,
                    lambdaInd, 
                    thisLambda,
                ]
            )
            print("\n")

            computeTrfs(
                parentDir,
                subject,
                presStopCorrections[iSubject],
                badChanLists[iSubject],
                condition,
                typeOfRegressor,
                nameOfRegressor,
                regressorDir,
                trialsToAnalyze=trialsToAnalyze,
                filenamePrefix=filenamePrefix,
                filenameSuffix=filenameSuffix,
                lambdaInd=lambdaInd, 
                thisLambda=thisLambda,
            )


# %%
for lambdaInd, thisLambda in enumerate(lambdas):
# for lambdaInd, thisLambda in enumerate(lambdas[-4:], start=len(lambdas) - 4):

    if doParallel:
    
        joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(joblib.delayed(doOneSubject)(
                iSubject, 
                subjects, 
                typeOfRegressors, 
                nameOfRegressors, 
                regressorDirs, 
                conditions, 
                filenamePrefix, 
                filenameSuffixes, 
                conditionNames, 
                parentDir, 
                presStopCorrections, 
                badChanLists, 
                lambdaInd, 
                thisLambda,
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
                regressorDirs, 
                conditions, 
                filenamePrefix, 
                filenameSuffixes, 
                conditionNames, 
                parentDir, 
                presStopCorrections, 
                badChanLists, 
                lambdaInd, 
                thisLambda,
            )




