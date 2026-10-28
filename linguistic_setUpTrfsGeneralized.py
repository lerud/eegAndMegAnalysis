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

from computeTrfsMEEG import computeTrfs

# %%
# parentDir = "/Users/karl/map/"
# parentDir = "/Volumes/Seagate/map/"
parentDir = "/scratch/zt1/project/jzsimon-prj/user/lerud/Volumes/Seagate/map/"

parallelType = "slurmArrayLambdasAndSubjects"  # or "joblibSubjects" or "serial"
n_jobs = 8
backend = "loky"
verbose = 49

lambdas = np.logspace(1, 14, 27)

slurmArrayInd = int(os.getenv("SLURM_ARRAY_TASK_ID"))

if parallelType == "slurmArrayLambdasAndSubjects":
    lambdaInd = int(slurmArrayInd % len(lambdas))
    iSubject = int(slurmArrayInd // len(lambdas))
else:
    lambdaInd = slurmArrayInd

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
    "R3265",
    "R3275",
    "R3247",
    "R3292",
    "R3293",
    "R3294",
    "R3303",
    "R3304",
    "R3306",
    "R3313",
    "R3317",
    "R3320",
    "R0286",
    "R3362",
]

subset = [i for i in range(len(subjects))]
# subset = [24, 25, 26]
subjects = [subjects[i] for i in subset]

ANmodelTackOn = [
    "-0",
    "-0",
    "-0",
    "-0",
    "-1",
    "-1",
    "-1",
    "-1",
    "-2",
    "-2",
    "-2",
    "-2",
    "-3",
    "-3",
    "-3",
    "-3",
    "-4",
    "-4",
    "-4",
    "-4",
    "-5",
    "-5",
    "-5",
    "-5",
    "-6",
    "-6",
    "-6",
    "-6",
    "-7",
    "-7",
    "-7",
]
ANmodelTackOn = [ANmodelTackOn[i] for i in subset]

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
    None,
    None,
    None,
]  # Remember the purpose of this is to tell us which stimulus was
# actually delivered to the subject: the ones with the ending
# trigger click (taking up an extra 1/600 seconds at the end), or
# the ones without the ending trigger (from before we had Triggy)

presStopCorrections = [presStopCorrections[i] for i in subset]

badChanLists = [
    [
        ["Fp2", "AF3", "FC1", "F3", "Fz", "Cz", "CP5", "P3", "CP6", "F7"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3045
    [
        ["P7", "CP6", "C4", "T7", "CP5", "P3", "P4", "O2", "Oz", "PO4", "F8"],
        ["MEG 056", "MEG 086"],
    ],  # R3089
    [["Oz", "P8", "CP6", "Fp2", "F8"], ["MEG 056", "MEG 086"]],  # R3093
    [["P7", "T8", "O2", "PO4", "Fp2", "F4", "T7"], ["MEG 056", "MEG 086"]],  # R3095
    [["C3", "FC5", "P4", "Oz"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3151
    [
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
            "P3",
            "FC1",
            "FC5",
            "T7",
            "AF4",
            "PO4",
        ],
        ["MEG 056", "MEG 086"],
    ],  # R2774
    [
        ["T7", "C3", "P7", "Pz", "O1", "P8", "CP6", "Oz", "O2"],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 062", "MEG 105"],
    ],  # R3152
    [
        ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 062"],
    ],  # R2877
    [
        ["CP6", "P8", "F4", "FC6", "T7", "C3", "Pz", "Oz", "O2", "P3"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3157
    [
        ["CP6", "P8", "T7", "CP5", "P7", "Oz", "FC5", "C3", "CP2"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R2783
    [
        ["CP5", "P3", "F3", "P4", "T7", "Fp1"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3170
    [
        [
            "F8",
            "FC6",
            "C4",
            "T8",
            "CP6",
            "P8",
            "P4",
            "P7",
            "Pz",
            "CP5",
            "FC1",
            "F7",
            "F3",
            "Oz",
        ],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 157"],
    ],  # R3172
    [None, ["MEG 056", "MEG 086", "MEG 046"]],  # R3193
    [
        ["F3", "T7", "F4", "FC6", "P8", "FC2", "P4", "P7", "FC2"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3184
    [["F8", "C3", "T8", "P7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3214
    [None, ["MEG 056", "MEG 086", "MEG 046", "MEG 062"]],  # R3265
    [
        ["T7", "P7", "F7", "T8"],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 157", "MEG 105"],
    ],  # R3275
    [["P4", "F8", "F3"], ["MEG 056", "MEG 086", "MEG 046", "MEG 157"]],  # R3247
    [["T8", "FC6"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3292
    [["Oz", "T7", "T8", "P3"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3293
    [
        ["CP6", "T8", "FC6", "T7", "FC5", "P7", "P8", "CP5", "F3", "Oz"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3294
    [["CP5", "P7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3303
    [["FC1"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3304
    [
        ["CP6", "T8", "P8", "PO4", "PO3", "O1", "O2", "FC6", "F4", "P7"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3306
    [
        [
            "CP6",
            "P8",
            "FC5",
            "FC6",
            "F3",
            "T7",
            "PO3",
            "PO4",
            "O1",
            "T8",
            "F4",
            "CP5",
            "C3",
            "FC1",
        ],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 121"],
    ],  # R3313
    [
        ["CP6", "T8", "P7", "P8", "Oz", "T7"],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 062"],
    ],  # R3317
    [
        ["P3", "F7", "FC5", "F8", "FC6", "F4", "T8"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3320
    [["PO3"], ["MEG 056", "MEG 086", "MEG 046", "MEG 087"]],  # R0286
    [["P8"], ["MEG 056", "MEG 086", "MEG 062", "MEG 087", "MEG 105"]],  # R3362
]
badChanLists = [badChanLists[i] for i in subset]

conditions = [
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "D",
    "A",
    "B",
    "C",
    "A",  # This is subject R0286
    "B",  # This is subject R3362
]
conditions = [conditions[i] for i in subset]

typeOfRegressors = [
    ["mix"],
    ["target"],
    ["target"],
    ["target"],
    ["distractor"],
    ["distractor"],
    ["distractor"],
    ["mix", "mix"],
    ["target"],
    ["target"],
    ["distractor"],
    ["distractor"],
    ["target"],
    ["target"],
    ["target"],
    ["distractor"],
    ["distractor"],
    ["distractor"],
    ["mix"],
    ["mix"],
    ["mix"],
    ["mix"],
    ["mix"],
    ["mix", "mix", "mix"],
    ["target", "target", "target", "target"],
    ["distractor", "distractor", "distractor", "distractor"],
    ["target", "target"],
    ["target", "target"],
    ["distractor", "distractor"],
    ["distractor", "distractor"],
    ["target", "target"],
    ["target", "target"],
    ["distractor", "distractor"],
    ["distractor", "distractor"],
]

regressorDirs = [
    [parentDir + "stimAndPredictors/mixes/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [
        parentDir + "stimAndPredictors/mixes/predictors/",
        parentDir + "stimAndPredictors/mixes/predictors/",
    ],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/targets/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/distractors/predictors/"],
    [parentDir + "stimAndPredictors/mixes/predictors/"],
    [parentDir + "stimAndPredictors/mixes/predictors/"],
    [parentDir + "stimAndPredictors/mixes/predictors/"],
    [parentDir + "stimAndPredictors/mixes/predictors/"],
    [parentDir + "stimAndPredictors/mixes/predictors/"],
    [
        parentDir + "stimAndPredictors/mixes/predictors/",
        parentDir + "stimAndPredictors/mixes/predictors/",
        parentDir + "stimAndPredictors/mixes/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/targets/predictors/",
        parentDir + "stimAndPredictors/targets/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
    ],
    [
        parentDir + "stimAndPredictors/distractors/predictors/",
        parentDir + "stimAndPredictors/distractors/predictors/",
    ],
]

filenameSuffixes = [
    ["_quiet"],
    ["_easy"],
    ["_hard"],
    ["_all"],
    ["_easy"],
    ["_hard"],
    ["_all"],
    ["_quiet_male", "_quiet_female"],
    ["_male"],
    ["_female"],
    ["_male"],
    ["_female"],
    ["_4dB"],
    ["_0dB"],
    ["_-4dB"],
    ["_4dB"],
    ["_0dB"],
    ["_-4dB"],
    ["_4dB"],
    ["_0dB"],
    ["_-4dB"],
    ["_easy"],
    ["_hard"],
    ["_ff_0dB", "_fm_0dB", "_mm_0dB"],
    ["_mf_easy", "_fm_easy", "_mf_hard", "_fm_hard"],
    ["_mf_easy", "_fm_easy", "_mf_hard", "_fm_hard"],
    ["_easy_male", "_easy_female"],
    ["_hard_male", "_hard_female"],
    ["_easy_male", "_easy_female"],
    ["_hard_male", "_hard_female"],
    ["_mm_easy", "_mm_hard"],
    ["_ff_easy", "_ff_hard"],
    ["_mm_easy", "_mm_hard"],
    ["_ff_easy", "_ff_hard"],
]

nameOfRegressors = [
    [
        "~wordOnsets_gaussian15msSD",
        "~phoneOnsets_gaussian15msSD",
        "~phsurp",
        "~cohtent",
        "~wordsurp",
    ]
]

filenamePrefix = "linguisticRecFieldAllFolds"


def doOneSubject(
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
    ANmodelTackOn,
):

    subject = subjects[iSubject]
    condition = conditions[iSubject]

    for nameOfRegressor in nameOfRegressors:

        for iType, typeOfRegressorAll in enumerate(typeOfRegressors):

            trialsToAnalyzeAll = []

            for parTypeInd in range(len(typeOfRegressorAll)):

                typeOfRegressor = typeOfRegressorAll[parTypeInd]
                regressorDir = regressorDirs[iType][parTypeInd]
                filenameSuffix = filenameSuffixes[iType][parTypeInd]

                trialsToAnalyze = []

                for iTrial in range(32):

                    conditionName = conditionNames[condition][iTrial]

                    if len(conditionName) > 16:

                        # t is quiet, y is easy, d is hard
                        if filenameSuffix == "_easy" and conditionName[-1] == "y":
                            trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_hard" and conditionName[-1] == "d":
                            trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_all":
                            trialsToAnalyze.append(
                                iTrial
                            )  # We're already inside if conditionName > 16, so all the rest are 2 speakers

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
                        if "_0dB" in filenameSuffix:
                            if (
                                conditionName[0] == "M"
                                and conditionName[16] == "M"
                                and conditionName[-1] == "d"
                                and "_fm" not in filenameSuffix
                                and "_ff" not in filenameSuffix
                            ):
                                trialsToAnalyze.append(iTrial)
                            elif (
                                conditionName[0] == "F"
                                and conditionName[16] == "F"
                                and conditionName[-1] == "d"
                                and "_fm" not in filenameSuffix
                                and "_mm" not in filenameSuffix
                            ):
                                trialsToAnalyze.append(iTrial)
                            elif (
                                conditionName[0] == "M"
                                and conditionName[16] == "F"
                                and conditionName[-1] == "y"
                                and "_mm" not in filenameSuffix
                                and "_ff" not in filenameSuffix
                            ):
                                trialsToAnalyze.append(iTrial)
                            elif (
                                conditionName[0] == "F"
                                and conditionName[16] == "M"
                                and conditionName[-1] == "y"
                                and "_mm" not in filenameSuffix
                                and "_ff" not in filenameSuffix
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
                        if filenameSuffix == "_mf_easy":
                            if (
                                conditionName[0] == "M"
                                and conditionName[16] == "F"
                                and conditionName[-1] == "y"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_fm_easy":
                            if (
                                conditionName[0] == "F"
                                and conditionName[16] == "M"
                                and conditionName[-1] == "y"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_mf_hard":
                            if (
                                conditionName[0] == "M"
                                and conditionName[16] == "F"
                                and conditionName[-1] == "d"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_fm_hard":
                            if (
                                conditionName[0] == "F"
                                and conditionName[16] == "M"
                                and conditionName[-1] == "d"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_easy_male":
                            if typeOfRegressor == "target":
                                if conditionName[0] == "M" and conditionName[-1] == "y":
                                    trialsToAnalyze.append(iTrial)
                            if typeOfRegressor == "distractor":
                                if (
                                    conditionName[16] == "M"
                                    and conditionName[-1] == "y"
                                ):
                                    trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_easy_female":
                            if typeOfRegressor == "target":
                                if conditionName[0] == "F" and conditionName[-1] == "y":
                                    trialsToAnalyze.append(iTrial)
                            if typeOfRegressor == "distractor":
                                if (
                                    conditionName[16] == "F"
                                    and conditionName[-1] == "y"
                                ):
                                    trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_hard_male":
                            if typeOfRegressor == "target":
                                if conditionName[0] == "M" and conditionName[-1] == "d":
                                    trialsToAnalyze.append(iTrial)
                            if typeOfRegressor == "distractor":
                                if (
                                    conditionName[16] == "M"
                                    and conditionName[-1] == "d"
                                ):
                                    trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_hard_female":
                            if typeOfRegressor == "target":
                                if conditionName[0] == "F" and conditionName[-1] == "d":
                                    trialsToAnalyze.append(iTrial)
                            if typeOfRegressor == "distractor":
                                if (
                                    conditionName[16] == "F"
                                    and conditionName[-1] == "d"
                                ):
                                    trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_mm_easy":
                            if (
                                conditionName[0] == "M"
                                and conditionName[16] == "M"
                                and conditionName[-1] == "y"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_mm_hard":
                            if (
                                conditionName[0] == "M"
                                and conditionName[16] == "M"
                                and conditionName[-1] == "d"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_ff_easy":
                            if (
                                conditionName[0] == "F"
                                and conditionName[16] == "F"
                                and conditionName[-1] == "y"
                            ):
                                trialsToAnalyze.append(iTrial)
                        if filenameSuffix == "_ff_hard":
                            if (
                                conditionName[0] == "F"
                                and conditionName[16] == "F"
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
                        if (
                            filenameSuffix == "_quiet_female"
                            and conditionName[0] == "F"
                        ):
                            trialsToAnalyze.append(iTrial)

                trialsToAnalyze = np.array(trialsToAnalyze)
                trialsToAnalyzeAll.append(trialsToAnalyze)

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

            joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
                joblib.delayed(computeTrfs)(
                    parentDir,
                    subject,
                    presStopCorrections[iSubject],
                    badChanLists[iSubject],
                    condition,
                    typeOfRegressorAll[parTypeInd],
                    nameOfRegressor,
                    regressorDirs[iType][parTypeInd],
                    trialsToAnalyze=trialsToAnalyzeAll[parTypeInd],
                    filenamePrefix=filenamePrefix,
                    filenameSuffix=filenameSuffixes[iType][parTypeInd],
                    lambdaInd=lambdaInd,
                    thisLambda=thisLambda,
                )
                for parTypeInd in range(len(typeOfRegressorAll))
            )


# %%
thisLambda = lambdas[lambdaInd]

if parallelType == "joblibSubjects":

    joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        joblib.delayed(doOneSubject)(
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
            ANmodelTackOn,
        )
        for iSubject in range(len(subjects))
    )

elif parallelType == "serial":

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
            ANmodelTackOn,
        )
elif parallelType == "slurmArrayLambdasAndSubjects":

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
        ANmodelTackOn,
    )

else:

    print("Wrong parallel type")
