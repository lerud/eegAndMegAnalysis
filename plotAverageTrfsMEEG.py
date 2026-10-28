# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: python3
# ---

# %%
import mne
import numpy as np
import eelbrain as eb
import scipy as sp
# import mat73
import matplotlib
matplotlib.use("QtAgg")
# # %matplotlib qt5
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pickle
import glob
import time
import gc
import copy

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *
from neuroAndSignalTools.utilityFuncs import calcPC
from prepTrfObjs import *

plt.ion()

# %%
# parentDir = "/Users/karl/map/"
parentDir = "/Volumes/Seagate/map/"


# filePrefix = "evoked"
# filePrefix = "recField"
# filePrefix = "concurRecField"
# filePrefix = "acousticConcurRecField"
# filePrefix = "acousticRecFieldAllFolds"
# filePrefix = "linguisticConcurRecField"
# filePrefix = "linguisticRecFieldAllFolds"
# filePrefix = "combinedRecFieldAllFolds"
# filePrefix = "temp_combinedRecFieldAllFolds"

filePrefix = "anMeegRecFieldAllFolds"
# filePrefix = "anRecFieldAllFolds"


pickleSuffix = ".pickle"
# pickleSuffix = ".pickle.CURRENT"
# pickleSuffix = ".pickle.LAPL"
# pickleSuffix = ".pickle.ORIG"
# pickleSuffix = ".pickle.WEIRD"

pickleSuffixForClicks = ".pickle"

# edgePad = 0.001

useDefaultMontage = True
# useDefaultMontage = False

if filePrefix == "anRecFieldAllFolds":
    subDirs = "/eegAndMeg/eeg/"
    nChannels = 32
    eegOnly = True
else:
    subDirs = "/eegAndMeg/meg/"
    nChannels = 32 + 157
    eegOnly = False

# shiftEvoked = None
shiftEvoked = "waveV"


# %% jp-MarkdownHeadingCollapsed=true
# nameOfRegressor = "~gammatone-1"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500

# nameOfRegressor = "~gammatone-on-1"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500

# nameOfRegressor = "~wordOnsets_gaussian15msSD"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500

# nameOfRegressor = "~phoneOnsets_gaussian15msSD"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500

# nameOfRegressor = "~phsurp"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500

# nameOfRegressor = "~cohtent"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500

# nameOfRegressor = "~wordsurp"
# lenResponse = 425
# lenToRamp = 0
# expToRamp = 2
# fs = 500


# nameOfRegressor = "_ANmodel_maxFs"
# lenResponse = 1395
# lenToRamp = .008
# expToRamp = 2
# fs = 16384


# nameOfRegressor = "_ANmodel_fs5000"
# lenResponse = 850
# lenToRamp = .02
# expToRamp = 4
# fs = 5000


nameOfRegressor = "_ANmodel_fs5000"
lenResponse = 500
lenToRamp = 0
expToRamp = 4
fs = 5000


# nameOfRegressor = "_glottalPulses_fs5000"
# lenResponse = 500
# lenToRamp = 0
# expToRamp = 4
# fs = 5000


# nameOfRegressor = "_halfWave_fs5000"
# lenResponse = 500
# lenToRamp = 0
# expToRamp = 4
# fs = 5000



# nameOfRegressor = "_glottalPulses_fs5000"
# lenResponse = 850
# lenToRamp = .02
# expToRamp = 4
# fs = 5000
# shiftEvoked = .00275






# nameOfRegressors = [
#     "~gammatone-1",
#     "~gammatone-on-1",
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordsurp",
# ]

# nameOfRegressors = [
#     "~gammatone-1",
#     "~gammatone-on-1",
# ]

# nameOfRegressors = [
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordsurp",
# ]



# nameOfRegressors = [
#     "_ANmodel_maxFs",
# ]


nameOfRegressors = [
    "_ANmodel_fs5000",
    "_glottalPulses_fs5000",
    "_halfWave_fs5000",
]


# %% jp-MarkdownHeadingCollapsed=true
optimizeLambda = False
plotLambdas = False

# lambdas = np.logspace(3, 11, 16)
# lambdaInd = 12  # This indexes the lambdas that have been run, defined above

# lambdas = np.logspace(1, 14, 27)
# lambdaInd = 20  # This indexes the lambdas that have been run, defined above

lambdas = np.array([0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20, 1e22])
lambdaInd = 0  # This indexes the lambdas that have been run, defined above

zScoring = True


timeToAddToStart = 0
# timeToAddToStart = 0.008

complexSubtypeNames = [" (real)", " (imag)", " (abs)", " (phase)"]

ci = 0.95  # confidence interval for shading



cutoffFreq = [40, 300]
filterTRFs = False
# sos = sp.signal.butter(3, cutoffFreq, output="sos")

normalizeByDataSTD = True

plotTopos = True
doSources = False
doWarpTesting = False
doPCplot = False
doPCplot2 = True
doClusters = True

subjectsToAverage = [
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

# additionalBadChansAll = [
#     None,  # R3045, 0
#     None,  # R3089, 1
#     None,  # R3093, 2
#     None,  # R3095, 3
#     None,  # R3151, 4
#     None,  # R2774, 5
#     None,  # R3152, 6
#     None,  # R2877, 7
#     None,  # R3157, 8
#     None,  # R2783, 9
#     None,  # R3170, 10
#     None,  # R3172, 11
#     None,  # R3193, 12
#     None,  # R3184, 13
#     None,  # R3214, 14
#     None,  # R3265, 15
#     ["T8"],  # R3275, 16
#     None,  # R3247, 17
#     ["T8", "FC6"],  # R3292, 18
#     None,  # R3293, 19
#     None,  # R3294, 20
#     None,  # R3303, 21
#     None,  # R3304, 22
#     None,  # R3306, 23
#     None,  # R3313, 24
#     None,  # R3317, 25
#     ["T8"],  # R3320, 26
# ]

additionalBadChansAll = None

clickList = ["R3265", "R3275", "R3247", "R3292", "R3293", "R3294", "R3303", "R3304", "R3306", "R3313", "R3317", "R3320", "R0286", "R3362"]

# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877", "R3157"]
# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877"]
# subjectsToAverage=["R3095", "R3151", "R2774", "R3152", "R2877", "R3157", "R2783"]
# subjectsToAverage=["R2877", "R3151", "R3152"]

# subjectsToAverage = ["R3093"]
# condIndForSources = 2

# subset = [i for i in range(len(subjectsToAverage)-2)]
# subset = [len(subjectsToAverage)-12]
# subset = np.arange(len(subjectsToAverage)-14, len(subjectsToAverage))
subset = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])  # Hard code these 12 subjects here, the group we're using for a manuscript
# subset = np.arange(27)  # The first 27 subjects, excluding R0286 and subsequent
# subset = np.delete(np.arange(27), [4, 5, 6, 7])  # The first 27 subjects, excluding R0286 and subsequent, and also excluding lab members
# subset = [subjectsToAverage.index("R3320")]

subjectsToAverage = [subjectsToAverage[i] for i in subset]

if additionalBadChansAll is not None:
    additionalBadChansAll = [additionalBadChansAll[i] for i in subset]

condIndForSources = 0

multFactor = 1

# cropEvoked = [0, .075]
# lenResponse = 375  # This is defined above, but overwrite it here if also defining cropEvoked; have to know it in advance

# This is what we've been using to go along with the source analysis of speech and click
cropEvoked = [0, .06875]
lenResponse = 345  # This is defined above, but overwrite it here if also defining cropEvoked; have to know it in advance

# cropEvoked = None

# eegOnly = True

avgOrAll = 0  # 0 for pre-averaged TRFs, 1 for lists of evoked objects for each subject

if len(subjectsToAverage) == 1 or avgOrAll == 1:
    resetAllBads = False
    nanOutBads = False
else:
    resetAllBads = True
    nanOutBads = True

# %%
rampFunction = np.concatenate(
    (
        np.linspace(0, 1, int(lenToRamp * fs)) ** expToRamp,
        np.ones(lenResponse - int(lenToRamp * fs)),
    )
)
rampFunction = rampFunction[None, :]

# %% jp-MarkdownHeadingCollapsed=true
typeOfRegressor = "mix"

filenameSuffix = "_quiet"

comment = "T: Quiet"

evokedAvgQuiet = loadAndPrepareEvokeds(typeOfRegressor, 
                              filenameSuffix, 
                              lenResponse, 
                              nChannels, 
                              subjectsToAverage, 
                              parentDir, 
                              subDirs, 
                              filePrefix, 
                              lambdas, 
                              lambdaInd, 
                              optimizeLambda, 
                              nameOfRegressor, 
                              nameOfRegressors,
                              nanOutBads,
                              zScoring,
                              comment,
                              complexSubtypeNames,
                              pickleSuffix,
                              shiftEvoked,
                              normalizeByDataSTD,
                              useDefaultMontage,
                              filterTRFs,
                              cutoffFreq,
                              timeToAddToStart,
                              rampFunction,
                              resetAllBads,
                              additionalBadChansAll,
                              multFactor=multFactor,
                              cropEvoked=cropEvoked,
                             )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_easy"

# comment = "T: Easy"

# evokedAvgTargetEasy = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_hard"

# comment = "T: Hard"

# evokedAvgTargetHard = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
typeOfRegressor = "target"

filenameSuffix = "_all"

comment = "T: All"

evokedAvgTargetAll = loadAndPrepareEvokeds(typeOfRegressor, 
                              filenameSuffix, 
                              lenResponse, 
                              nChannels, 
                              subjectsToAverage, 
                              parentDir, 
                              subDirs, 
                              filePrefix, 
                              lambdas, 
                              lambdaInd, 
                              optimizeLambda, 
                              nameOfRegressor, 
                              nameOfRegressors,
                              nanOutBads,
                              zScoring,
                              comment,
                              complexSubtypeNames,
                              pickleSuffix,
                              shiftEvoked,
                              normalizeByDataSTD,
                              useDefaultMontage,
                              filterTRFs,
                              cutoffFreq,
                              timeToAddToStart,
                              rampFunction,
                              resetAllBads,
                              additionalBadChansAll,
                              multFactor=multFactor,
                              cropEvoked=cropEvoked,
                             )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_easy"

# comment = "D: Easy"

# evokedAvgDistEasy = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_hard"

# comment = "D: Hard"

# evokedAvgDistHard = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
typeOfRegressor = "distractor"

filenameSuffix = "_all"

comment = "D: All"

evokedAvgDistAll = loadAndPrepareEvokeds(typeOfRegressor, 
                              filenameSuffix, 
                              lenResponse, 
                              nChannels, 
                              subjectsToAverage, 
                              parentDir, 
                              subDirs, 
                              filePrefix, 
                              lambdas, 
                              lambdaInd, 
                              optimizeLambda, 
                              nameOfRegressor, 
                              nameOfRegressors,
                              nanOutBads,
                              zScoring,
                              comment,
                              complexSubtypeNames,
                              pickleSuffix,
                              shiftEvoked,
                              normalizeByDataSTD,
                              useDefaultMontage,
                              filterTRFs,
                              cutoffFreq,
                              timeToAddToStart,
                              rampFunction,
                              resetAllBads,
                              additionalBadChansAll,
                              multFactor=multFactor,
                              cropEvoked=cropEvoked,
                             )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_male"

# comment = "T: Male"

# evokedAvgTargetMale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_female"

# comment = "T: Female"

# evokedAvgTargetFemale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_quiet_male"

# comment = "T: Quiet male"

# evokedAvgQuietMale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_quiet_female"

# comment = "T: Quiet female"

# evokedAvgQuietFemale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_male"

# comment = "D: Male"

# evokedAvgDistMale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_female"

# comment = "D: Female"

# evokedAvgDistFemale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_4dB"

# comment = "M: 4 dB"

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_0dB"

# # comment = "M: 0 dB"
# comment = "M: All"  # Make the evoked object comment the same as what the helper functions are expecting in order to get the wave V times from what are already calculated

# evokedAvgMix0dB = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_-4dB"

# comment = "M: -4 dB"

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_4dB"

# comment = "T: 4 dB"

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_0dB"

# # comment = "T: 0 dB"
# comment = "T: All"  # Make the evoked object comment the same as what the helper functions are expecting in order to get the wave V times from what are already calculated

# evokedAvgTarget0dB = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_-4dB"

# comment = "T: -4 dB"

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_4dB"

# comment = "D: 4 dB"

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_0dB"

# # comment = "D: 0 dB"
# comment = "D: All"  # Make the evoked object comment the same as what the helper functions are expecting in order to get the wave V times from what are already calculated

# evokedAvgDist0dB = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_-4dB"

# comment = "D: -4 dB"

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_easy"

# comment = "M: Easy"

# evokedAvgMixEasy = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
typeOfRegressor = "mix"

filenameSuffix = ["_easy", "_hard"]

comment = "M: All"

evokedAvgMixAll = loadAndPrepareEvokeds(typeOfRegressor, 
                              filenameSuffix, 
                              lenResponse, 
                              nChannels, 
                              subjectsToAverage, 
                              parentDir, 
                              subDirs, 
                              filePrefix, 
                              lambdas, 
                              lambdaInd, 
                              optimizeLambda, 
                              nameOfRegressor, 
                              nameOfRegressors,
                              nanOutBads,
                              zScoring,
                              comment,
                              complexSubtypeNames,
                              pickleSuffix,
                              shiftEvoked,
                              normalizeByDataSTD,
                              useDefaultMontage,
                              filterTRFs,
                              cutoffFreq,
                              timeToAddToStart,
                              rampFunction,
                              resetAllBads,
                              additionalBadChansAll,
                              multFactor=multFactor,
                              cropEvoked=cropEvoked,
                             )

# %%
# typeOfRegressor = "target"

# filenameSuffix = ["_mf_easy", "_fm_easy"]

# comment = "T: Mixed Easy"

# evokedAvgTargetMixedEasy = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "target"

# filenameSuffix = [ "_mm_hard", "_ff_hard"]

# comment = "T: Same Hard"

# evokedAvgTargetSameHard = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "target"

# filenameSuffix = ["_mm_easy", "_ff_easy"]

# comment = "T: Same Easy"

# evokedAvgTargetSameEasy = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "target"

# filenameSuffix = ["_mf_hard", "_fm_hard"]

# comment = "T: Mixed Hard"

# evokedAvgTargetMixedHard = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = ["_mf_easy", "_fm_easy"]

# comment = "D: Mixed Easy"

# evokedAvgDistractorMixedEasy = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = ["_mm_hard", "_ff_hard"]

# comment = "D: Same Hard"

# evokedAvgDistractorSameHard = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_hard"

# comment = "M: Hard"

# evokedAvgMixHard = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_ff_0dB"

# comment = "M: FF 0 dB"

# evokedAvgMixFF0dB = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_fm_0dB"

# comment = "M: FM 0 dB"

# evokedAvgMixFM0dB = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_mm_0dB"

# comment = "M: MM 0 dB"

# evokedAvgMixMM0dB = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                              )

# %%
# typeOfRegressor = "target"

# # filenameSuffix = ["_mf_easy", "_mm_hard", "_fm_easy", "_ff_hard"]
# filenameSuffix = ["_mf_easy", "_mm_hard"]

# # comment = "T: 0 dB"
# comment = "T: All"  # Make the evoked object comment the same as what the helper functions are expecting in order to get the wave V times from what are already calculated

# evokedAvgTargetMale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
# typeOfRegressor = "distractor"

# # filenameSuffix = ["_fm_easy", "_mm_hard", "_mf_easy", "_ff_hard"]
# filenameSuffix = ["_fm_easy", "_mm_hard"]

# # comment = "T: 0 dB"
# comment = "D: All"  # Make the evoked object comment the same as what the helper functions are expecting in order to get the wave V times from what are already calculated

# evokedAvgDistMale = loadAndPrepareEvokeds(typeOfRegressor, 
#                               filenameSuffix, 
#                               lenResponse, 
#                               nChannels, 
#                               subjectsToAverage, 
#                               parentDir, 
#                               subDirs, 
#                               filePrefix, 
#                               lambdas, 
#                               lambdaInd, 
#                               optimizeLambda, 
#                               nameOfRegressor, 
#                               nameOfRegressors,
#                               nanOutBads,
#                               zScoring,
#                               comment,
#                               complexSubtypeNames,
#                               pickleSuffix,
#                               shiftEvoked,
#                               normalizeByDataSTD,
#                               useDefaultMontage,
#                               filterTRFs,
#                               cutoffFreq,
#                               timeToAddToStart,
#                               rampFunction,
#                               resetAllBads,
#                               additionalBadChansAll,
#                               multFactor=multFactor,
#                               cropEvoked=cropEvoked,
#                              )

# %%
if set(subjectsToAverage).issubset(set(clickList)) and filePrefix[:2] == "an":
    # Redefine some of these variables here because they were defined at the top for speech maintask responses
    filePrefix = "abrAndTones"
    nameOfRegressor = "_ANmodel_clicks"
    # lenResponse = 500
    lenToRamp = 0
    expToRamp = 4
    fs = 5000
    lambdaIndForClicks = lambdaInd
    # lambdaIndForClicks = 0
    nameOfRegressors = [
        "_ANmodel_clicks",
    ]
    typeOfRegressor = "clicksComplete"

    comment = "T: Clicks complete"

    filenameSuffix = "_minimumPhase"

    evokedAvgClicksComplete = loadAndPrepareEvokeds(typeOfRegressor, 
                                  filenameSuffix, 
                                  lenResponse, 
                                  nChannels, 
                                  subjectsToAverage, 
                                  parentDir, 
                                  subDirs, 
                                  filePrefix, 
                                  lambdas, 
                                  lambdaIndForClicks, 
                                  optimizeLambda, 
                                  nameOfRegressor, 
                                  nameOfRegressors,
                                  nanOutBads,
                                  zScoring,
                                  comment,
                                  complexSubtypeNames,
                                  pickleSuffixForClicks,
                                  shiftEvoked,
                                  normalizeByDataSTD,
                                  useDefaultMontage,
                                  filterTRFs,
                                  cutoffFreq,
                                  timeToAddToStart,
                                  rampFunction,
                                  resetAllBads,
                                  additionalBadChansAll,
                                  multFactor=multFactor,
                                  cropEvoked=cropEvoked,
                                 )

# %%
linewidth = 3
alpha = 0.75

complexSubtype = 0



# condsToPlot = [
#     evokedAvgQuiet,
# ]
# colors = ["blue"]




# condsToPlot = [
#     evokedAvgQuiet,
#     evokedAvgTargetEasy,
#     evokedAvgTargetHard,
#     evokedAvgDistHard,
#     evokedAvgDistEasy,
# ]
# colors = ["blue",  "cyan", "green", "brown", "red"]



# condsToPlot = [
#     evokedAvgQuiet,
#     evokedAvgTargetAll,
#     evokedAvgDistAll,
# ]
# colors = ["blue", "green", "red"]



# condsToPlot = [evokedAvgDistHard, evokedAvgTargetEasy, evokedAvgDistEasy, evokedAvgTargetHard]
# colors = ["brown", "cyan", "red", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTargetEasy, evokedAvgTargetHard]
# colors = ["blue", "cyan", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgMixHard, evokedAvgMixEasy]
# colors = ["blue", "red", "green"]

        

# if set(subjectsToAverage).issubset(set(clickList)):# and filePrefix[:2] == "an":
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll, evokedAvgClicksComplete]
#     colors = ["blue", "green", "red", "purple", "black"]
# else:
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll]
#     colors = ["blue", "green", "red", "purple"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll]
# colors = ["blue", "green", "red", "purple"]




        

# if set(subjectsToAverage).issubset(set(clickList)):# and filePrefix[:2] == "an":
#     condsToPlot = [evokedAvgQuiet, evokedAvgMixAll, evokedAvgClicksComplete]
#     colors = ["blue", "purple", "black"]
# else:
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll]
#     colors = ["blue", "green", "red", "purple"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll]
# colors = ["blue", "green", "red", "purple"]




        

if set(subjectsToAverage).issubset(set(clickList)):# and filePrefix[:2] == "an":
    condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll, evokedAvgClicksComplete]
    colors = ["black", "purple", "green", "blue", "orange"]
else:
    condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll]
    colors = ["blue", "green", "red", "purple", "orange"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixAll]
# colors = ["black", "purple", "green", "blue", ]




# condsToPlot = [evokedAvgTargetMixedEasy, evokedAvgTargetSameHard, evokedAvgDistractorMixedEasy, evokedAvgDistractorSameHard]
# colors = ["cyan", "green", "brown", "red"]




# condsToPlot = [evokedAvgTargetMixedEasy, evokedAvgTargetMixedHard, evokedAvgTargetSameEasy, evokedAvgTargetSameHard]
# colors = ["xkcd:blue", "xkcd:light blue", "xkcd:green", "xkcd:light green"]
# colors = ["xkcd:dark red", "xkcd:red", "xkcd:orange", "xkcd:light orange"]



        

# if set(subjectsToAverage).issubset(set(clickList)):
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetMale, evokedAvgDistMale, evokedAvgMixAll, evokedAvgClicksComplete]
#     colors = ["blue", "green", "red", "purple", "black"]
# else:
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetMale, evokedAvgDistMale, evokedAvgMixAll]
#     colors = ["blue", "green", "red", "purple"]



        

        

# condsToPlot = [evokedAvgTargetMale, evokedAvgTargetFemale, evokedAvgDistMale, evokedAvgDistFemale]
# colors = ["cyan", "green", "red", "brown"]

        

# condsToPlot = [evokedAvgTargetEasy, evokedAvgTargetHard, evokedAvgDistEasy, evokedAvgDistHard]
# colors = ["cyan", "green", "red", "brown"]


        

# if set(subjectsToAverage).issubset(set(clickList)):
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixEasy, evokedAvgMixHard, evokedAvgClicksComplete]
#     colors = ["blue", "green", "red", "purple", "pink", "black"]
# else:
#     condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixEasy, evokedAvgMixHard]
#     colors = ["blue", "green", "red", "purple", "pink"]


# condsToPlot = [evokedAvgQuiet, evokedAvgMix4dB, evokedAvgMix0dB, evokedAvgMixNeg4dB]
# colors = ["blue", "cyan", "xkcd:light green", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgMix4dB, evokedAvgMix0dB, evokedAvgMixNeg4dB, evokedAvgDistHard]
# colors = ["blue", "cyan", "xkcd:light green", "green", "brown"]

# condsToPlot = [evokedAvgMix4dB, evokedAvgMix0dB, evokedAvgMixNeg4dB]
# colors = ["cyan", "xkcd:light green", "green"]

# condsToPlot = [evokedAvgMixEasy, evokedAvgMixHard]
# colors = ["cyan", "red"]


# condsToPlot = [evokedAvgTarget4dB, evokedAvgTargetNeg4dB, evokedAvgDist4dB, evokedAvgDistNeg4dB]
# colors = ["blue", "green", "red", "brown"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTarget4dB, evokedAvgTargetNeg4dB, evokedAvgDist4dB, evokedAvgDistNeg4dB]
# colors = ["blue", "cyan", "green", "red", "brown"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTarget4dB, evokedAvgTarget0dB, evokedAvgTargetNeg4dB]
# colors = ["blue", "cyan", "xkcd:light green", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgDist4dB, evokedAvgDist0dB, evokedAvgDistNeg4dB]
# colors = ["blue", "pink", "red", "brown"]


# condsToPlot = [evokedAvgQuiet, evokedAvgQuietMale, evokedAvgQuietFemale]
# colors = ["blue", "xkcd:blue", "xkcd:green"]

# condsToPlot = [evokedAvgQuietMale, evokedAvgQuietFemale]
# colors = ["blue", "green"]

# condsToPlot = [evokedAvgTargetMale, evokedAvgTargetFemale]
# colors = ["blue", "green"]

# condsToPlot = [evokedAvgTargetMale, evokedAvgTargetFemale, evokedAvgDistMale, evokedAvgDistFemale]
# colors = ["blue", "green", "red", "brown"]

# condsToPlot = [evokedAvgQuietMale, evokedAvgTargetMale, evokedAvgQuietFemale, evokedAvgTargetFemale]
# colors = ["xkcd:blue", "xkcd:light blue", "xkcd:green", "xkcd:light green"]


# condsToPlot = [evokedAvgQuiet, evokedAvgMixFF0dB, evokedAvgMixFM0dB, evokedAvgMixMM0dB]
# colors = ["blue", "green", "xkcd:blue", "xkcd:green"]




# condsToPlot = [evokedAvgQuietMale, evokedAvgQuietFemale, evokedAvgMixFF0dB, evokedAvgMixFM0dB, evokedAvgMixMM0dB]
# colors = ["blue", "green", "cyan", "red", "brown"]





keys = [cond[0][complexSubtype].comment for cond in condsToPlot]
evokedValues = [cond[avgOrAll][complexSubtype] for cond in condsToPlot]
evokeds = {k: v for (k, v) in zip(keys, evokedValues)}

styleValues = [
    {"linewidth": linewidth, "alpha": alpha, "color": colors[i]}
    for i in range(len(condsToPlot))
]
styles = {k: v for (k, v) in zip(keys, styleValues)}

if plotTopos:
    mne.viz.plot_compare_evokeds(
        evokeds=evokeds, legend=True, axes="topo", styles=styles, ci=ci, picks="eeg"
    )


# %%
if plotTopos:
    if not eegOnly:
        mne.viz.plot_compare_evokeds(
            evokeds=evokeds, legend=True, axes="topo", styles=styles, ci=ci, picks="meg"
        )

# %%
if filePrefix[:3] == "anR":
    peakRanges = [[.007, .0155], [.023, .04], [.04, .06]]
    numPeaksPerRange = [2, 2, 1]
    useEEGonly = [[True, True], [False, False], [False]]
    useMEGonly = [[False, False], [False, False], [False]]
    order = 30
# elif filePrefix[:3] == "anM":
#     peakRanges = [[.007, .0145], [.0245, .031], [.04, .06], [.08, .12]]
#     numPeaksPerRange = [2, 1, 1, 1]
#     useEEGonly = [[True, True], [True], [False], [False]]
#     useMEGonly = [[False, False], [False], [False], [False]]
#     order = 5
# elif filePrefix[:3] == "anM":
#     peakRanges = [[.007, .0145], [.0245, .031], [.04, .06]]
#     numPeaksPerRange = [2, 1, 1]
#     useEEGonly = [[True, True], [True], [False]]
#     useMEGonly = [[False, False], [False], [False]]
#     order = 5
elif filePrefix[:3] == "anM" or filePrefix[:3] == "abr":
    peakRanges = [[.0074, .0106],[.01, .0158], [.0245, .031], [.045, .06]]
    numPeaksPerRange = [1, 1, 1, 1]
    useEEGonly = [[True], [True], [True], [False]]
    eegSigns = [None, None, None, None]
    useMEGonly = [[False], [False], [False], [False]]
    order = 5
else:
    peakRanges = [[.015, .07], [.07, .23], [.25, .7]]
    numPeaksPerRange = [1, 2, 2]
    useEEGonly = [[False], [False, False], [False, False]]
    eegSigns = [None, None, None]
    useMEGonly = [[True], [True, True], [True, True]]
    # peakRanges = [[.015, .07], [.08, .12], [.13, .23]]
    # numPeaksPerRange = [1, 1, 1]
    # useEEGonly = [[True], [True], [True]]
    order = 10

plotTimes = False
timesFive = createPeakTimes(avgOrAll, condsToPlot, peakRanges, numPeaksPerRange, useEEGonly, useMEGonly, order, plotTimes, eegOnly, eegSigns)
print(timesFive)


# %%
if plotTopos:
    if filePrefix[:3] == "anR":
        peakRanges = [[.0002, .0155], [.017, .04], [.04, .06]]
        numPeaksPerRange = [4, 3, 1]
        useEEGonly = [[True, True, True, False], [False, False, False], [False]]
        eegSigns = [None, None, None]
        useMEGonly = [[False, False, False, False], [False, False, False], [False]]
        order = 30
    elif filePrefix[:3] == "anM" or filePrefix[:3] == "abr":
        peakRanges = [[.0002, .0155], [.017, .04], [.04, .06]]
        numPeaksPerRange = [4, 2, 2]
        useEEGonly = [[True, True, True, True], [False, False], [False, False]]
        eegSigns = [None, None, None]
        useMEGonly = [[False, False, False, False], [True, True], [False, False]]
        order = 5
    else:
        peakRanges = [[.006, .01], [.01, .014], [.015, .07], [.07, .25], [.25, .7]]
        numPeaksPerRange = [1, 1, 1, 1, 2, 2]
        useEEGonly = [[False], [False], [False], [False], [False, False], [False, False]]
        eegSigns = [None, None, None, None, None]
        useMEGonly = [[False], [False], [True], [True], [True, True], [True, True]]
        # peakRanges = [[.015, .07], [.08, .12], [.13, .23]]
        # numPeaksPerRange = [1, 1, 1]
        # useEEGonly = [[True], [True], [True]]
        order = 10
    
    plotTimes = False
    timesEight = createPeakTimes(avgOrAll, condsToPlot, peakRanges, numPeaksPerRange, useEEGonly, useMEGonly, order, plotTimes, eegOnly, eegSigns)

# %%
custLines = True
jointLinewidth = 2
jointAlpha = .75
perMapVlims = True
figSizeJoint = (10, 5)
saveJointFigs = False
# plt.style.use("dark_background")

timesFiveEEG = copy.deepcopy(timesFive)
timesFiveMEG = copy.deepcopy(timesFive)

# # Comment this block out to not mess with the calculated time points
# timesFiveMEG[3] = np.array([0.008, 0.0109, 0.027, 0.0345, 0.0504])
# timesFiveEEG[3] = np.array([0.0086, 0.011, 0.027, 0.034, 0.0465])
# timesFiveMEG[4] = np.array([0.0075, 0.0154, 0.029, 0.043, 0.055])
# timesFiveEEG[4] = np.array([0.0084, 0.0154, 0.029, 0.041, 0.0532])

for i, evoked in enumerate(evokedValues):
    figJointEEG = evoked.plot_joint(picks="eeg", times=timesFiveEEG[i], title=evoked.comment, ts_args={"gfp": True, "time_unit": "ms"}, topomap_args={"average": 0.002, "time_unit": "ms"})
    if custLines:
        for line in figJointEEG.axes[len(timesFiveEEG[i])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
    if perMapVlims:
        for j, axis in enumerate(figJointEEG.axes[:len(timesFiveEEG[i])]):
            image = axis.images[0]
            tempData = evoked.copy().crop(tmin=timesFiveEEG[i][j], tmax=timesFiveEEG[i][j]).get_data(picks="eeg")
            cmapRange = np.max(np.abs([np.min(tempData), np.max(tempData)])) * 1e6
            image.set_clim((-cmapRange, cmapRange))
    figJointEEG.set_size_inches(figSizeJoint)
    if saveJointFigs:
        figJointEEG.savefig(f"/Users/karl/Dropbox/UMD/ARO2026/talk/eegJoint_{i}.pdf")
            
    if not eegOnly:
        figJointMEG = evoked.plot_joint(picks="meg", times=timesFiveMEG[i], title=evoked.comment, ts_args={"gfp": True, "time_unit": "ms"}, topomap_args={"average": 0.002, "time_unit": "ms"})
        if custLines:
            for line in figJointMEG.axes[len(timesFiveMEG[i])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
                line.set_linewidth(jointLinewidth)
                line.set_alpha(jointAlpha)
        if perMapVlims:
            for j, axis in enumerate(figJointMEG.axes[:len(timesFiveMEG[i])]):
                image = axis.images[0]
                tempData = evoked.copy().crop(tmin=timesFiveMEG[i][j], tmax=timesFiveMEG[i][j]).get_data(picks="meg")
                cmapRange = np.max(np.abs([np.min(tempData), np.max(tempData)])) * 1e15
                image.set_clim((-cmapRange, cmapRange))
        figJointMEG.set_size_inches(figSizeJoint)
        if saveJointFigs:
            figJointMEG.savefig(f"/Users/karl/Dropbox/UMD/ARO2026/talk/megJoint_{i}.pdf")

# %%
if plotTopos:

    condTopoInd = 0
    windowMultiplier = 1
    
    timesEeg = timesEight[condTopoInd]
    timesMeg = timesEeg
    averageEeg = timesEeg / 10
    averageMeg = averageEeg
    
    figEeg, axsEeg = plt.subplots(2, 4, figsize=[20, 10])
    for i in range(len(timesEeg)):
        evokedValues[condTopoInd].plot_topomap(times=timesEeg[i], average=averageEeg[i], ch_type="eeg", axes=axsEeg[i // axsEeg.shape[1], i % axsEeg.shape[1]], colorbar=False)
    evokedValues[condTopoInd].plot_topomap(times=timesEeg, average=averageEeg, ch_type="eeg")
    figEeg.suptitle(f"EEG for lambda {lambdas[lambdaInd]}")
    
    if not eegOnly:
        figMeg, axsMeg = plt.subplots(2, 4, figsize=[20, 10])
        for i in range(len(timesMeg)):
            evokedValues[condTopoInd].plot_topomap(times=timesMeg[i], average=averageMeg[i], ch_type="mag", axes=axsMeg[i // axsMeg.shape[1], i % axsMeg.shape[1]], colorbar=False)
        evokedValues[condTopoInd].plot_topomap(times=timesMeg, average=averageMeg, ch_type="mag")
        figMeg.suptitle(f"MEG for lambda {lambdas[lambdaInd]}")

# %%
if plotTopos:

    if not eegOnly:
    
        # fig1 = plt.figure(figsize=[18,5])
    
        # cmap = "RdBu_r"
        cmap = "Reds"
    
        # for i in range(1, 1+len(condsToPlot)):
    
        #     ax = fig1.add_subplot(1, len(condsToPlot), i)
    
        #     im, cn = mne.viz.plot_topomap(condsToPlot[2][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax)
        #     ax.colorbar(im)
    
        maxAllEEG = 0
        minAllEEG = 0
        for i in range(len(condsToPlot)):
            maxAllEEG = np.array([maxAllEEG, condsToPlot[i][2][157:189].max()]).max()
            minAllEEG = np.array([minAllEEG, condsToPlot[i][2][157:189].min()]).min()
    
        # fig2, (ax5, ax6, ax7, ax8, ax9) = plt.subplots(ncols=5, figsize=[21, 6])
        fig2, axes2 = plt.subplots(ncols=len(condsToPlot), figsize=[14, 4])
    
        for i in range(len(condsToPlot)):
            
            im, cn = mne.viz.plot_topomap(
                condsToPlot[i][2][157:189],
                mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)),
                cmap=cmap,
                axes=axes2[i],
                vlim=(minAllEEG, maxAllEEG),
            )
            axes2[i].set_title(condsToPlot[i][0][0].comment)
            # Create a new axes for the colorbar
            # cax2 = fig2.add_axes([0.92, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
        
            # Add the colorbar to the new axes
            # fig2.colorbar(im, cax=axes2[i])
        
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[0][2][157:189],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)),
        #     cmap=cmap,
        #     axes=ax5,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax5.set_title(condsToPlot[0][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[2][2][157:189],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)),
        #     cmap=cmap,
        #     axes=ax6,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax6.set_title(condsToPlot[2][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[4][2][157:189],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)),
        #     cmap=cmap,
        #     axes=ax7,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax7.set_title(condsToPlot[4][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[1][2][157:189],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)),
        #     cmap=cmap,
        #     axes=ax8,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax8.set_title(condsToPlot[1][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[3][2][157:189],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)),
        #     cmap=cmap,
        #     axes=ax9,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax9.set_title(condsToPlot[3][0][0].comment)
        # im,cm   = mne.viz.plot_topomap(x2, info, axes=ax2,show=False,vmin=vmin,vmax=vmax)
        # manually fiddle the position of colorbar
        # ax_x_start = 0.95
        # ax_x_width = 0.04
        # ax_y_start = 0.1
        # ax_y_height = 0.9
        # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        # clb = fig.colorbar(im, cax=cbar_ax)
        # clb2 = fig2.colorbar(im)
    




        # Create a new axes for the colorbar
        cax2 = fig2.add_axes([0.92, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
    
        # Add the colorbar to the new axes
        fig2.colorbar(im, cax=cax2, label="r (correlation coefficient)")
    
        fig2.suptitle(f"Lambda of {lambdas[lambdaInd]}, max Pearson r of {maxAllEEG}")





    
        print(f"Max for EEG is {maxAllEEG}")
    
        # clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar = plt.figure(figsize=[18,5])
    
    
    else:

        cmap = "RdBu_r"
        # cmap="Reds"
    
        # for i in range(1, 1+len(condsToPlot)):
    
        #     ax = fig1.add_subplot(1, len(condsToPlot), i)
    
        #     im, cn = mne.viz.plot_topomap(condsToPlot[2][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax)
        #     ax.colorbar(im)
    
        maxAllEEG = 0
        minAllEEG = 0
        for i in range(len(condsToPlot)):
            maxAllEEG = np.array([maxAllEEG, condsToPlot[i][2].max()]).max()
            minAllEEG = np.array([minAllEEG, condsToPlot[i][2].min()]).min()
    
        fig2, axes2 = plt.subplots(ncols=5, figsize=[21, 6])
    
        for i in range(len(condsToPlot)):
        
            im, cn = mne.viz.plot_topomap(
                condsToPlot[i][2],
                mne.pick_info(evokedValues[0].info, sel=None),
                cmap=cmap,
                axes=axes2[i],
                vlim=(minAllEEG, maxAllEEG),
            )
            axes2[i].set_title(condsToPlot[i][0][0].comment)

        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[0][2],
        #     mne.pick_info(evokedValues[0].info, sel=None),
        #     cmap=cmap,
        #     axes=ax5,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax5.set_title(condsToPlot[0][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[2][2],
        #     mne.pick_info(evokedValues[0].info, sel=None),
        #     cmap=cmap,
        #     axes=ax6,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax6.set_title(condsToPlot[2][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[4][2],
        #     mne.pick_info(evokedValues[0].info, sel=None),
        #     cmap=cmap,
        #     axes=ax7,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax7.set_title(condsToPlot[4][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[1][2],
        #     mne.pick_info(evokedValues[0].info, sel=None),
        #     cmap=cmap,
        #     axes=ax8,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax8.set_title(condsToPlot[1][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[3][2],
        #     mne.pick_info(evokedValues[0].info, sel=None),
        #     cmap=cmap,
        #     axes=ax9,
        #     vlim=(minAllEEG, maxAllEEG),
        # )
        # ax9.set_title(condsToPlot[3][0][0].comment)
        # im,cm   = mne.viz.plot_topomap(x2, info, axes=ax2,show=False,vmin=vmin,vmax=vmax)
        # manually fiddle the position of colorbar
        # ax_x_start = 0.95
        # ax_x_width = 0.04
        # ax_y_start = 0.1
        # ax_y_height = 0.9
        # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        # clb = fig.colorbar(im, cax=cbar_ax)
        # clb2 = fig2.colorbar(im)
    
        # Create a new axes for the colorbar
        cax2 = fig2.add_axes([0.92, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
    
        # Add the colorbar to the new axes
        fig2.colorbar(im, cax=cax2)
    
        fig2.suptitle(f"Lambda of {lambdas[lambdaInd]}, max Pearson r of {maxAllEEG}")
    
        print(f"Max for EEG is {maxAllEEG}")
    
        # clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar

# %%
if plotTopos:

    if not eegOnly:
    
        # fig1 = plt.figure(figsize=[18,5])
    
        # cmap = "RdBu_r"
        cmap="Reds"
    
        # for i in range(1, 1+len(condsToPlot)):
    
        #     ax = fig1.add_subplot(1, len(condsToPlot), i)
    
        #     im, cn = mne.viz.plot_topomap(condsToPlot[2][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax)
        #     ax.colorbar(im)
    
        maxAllMEG = 0
        minAllMEG = 0
        for i in range(len(condsToPlot)):
            maxAllMEG = np.array([maxAllMEG, condsToPlot[i][2][0:157].max()]).max()
            minAllMEG = np.array([minAllMEG, condsToPlot[i][2][0:157].min()]).min()
    
        # fig1, axes1 = plt.subplots(ncols=5, figsize=[21, 6])
        fig1, axes1 = plt.subplots(ncols=len(condsToPlot), figsize=[14, 4])
        
        for i in range(len(condsToPlot)):
    
            im, cn = mne.viz.plot_topomap(
                condsToPlot[i][2][0:157],
                mne.pick_info(evokedValues[0].info, sel=np.arange(157)),
                cmap=cmap,
                axes=axes1[i],
                vlim=(minAllMEG, maxAllMEG),
            )
            axes1[i].set_title(condsToPlot[i][0][0].comment)
    
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[0][2][0:157],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157)),
        #     cmap=cmap,
        #     axes=ax0,
        #     vlim=(minAllMEG, maxAllMEG),
        # )
        # ax0.set_title(condsToPlot[0][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[2][2][0:157],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157)),
        #     cmap=cmap,
        #     axes=ax1,
        #     vlim=(minAllMEG, maxAllMEG),
        # )
        # ax1.set_title(condsToPlot[2][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[4][2][0:157],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157)),
        #     cmap=cmap,
        #     axes=ax2,
        #     vlim=(minAllMEG, maxAllMEG),
        # )
        # ax2.set_title(condsToPlot[4][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[1][2][0:157],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157)),
        #     cmap=cmap,
        #     axes=ax3,
        #     vlim=(minAllMEG, maxAllMEG),
        # )
        # ax3.set_title(condsToPlot[1][0][0].comment)
        # im, cn = mne.viz.plot_topomap(
        #     condsToPlot[3][2][0:157],
        #     mne.pick_info(evokedValues[0].info, sel=np.arange(157)),
        #     cmap=cmap,
        #     axes=ax4,
        #     vlim=(minAllMEG, maxAllMEG),
        # )
        # ax4.set_title(condsToPlot[3][0][0].comment)
        # im,cm   = mne.viz.plot_topomap(x2, info, axes=ax2,show=False,vmin=vmin,vmax=vmax)
        # manually fiddle the position of colorbar
        # ax_x_start = 0.95
        # ax_x_width = 0.04
        # ax_y_start = 0.1
        # ax_y_height = 0.9
        # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        # clb = fig.colorbar(im, cax=cbar_ax)
    
        # clb1 = fig1.colorbar(im)
    
        # Create a new axes for the colorbar
        cax1 = fig1.add_axes([0.92, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
    
        # Add the colorbar to the new axes
        fig1.colorbar(im, cax=cax1, label="r (correlation coefficient)")
    
        fig1.suptitle(f"Lambda of {lambdas[lambdaInd]}, max Pearson r of {maxAllMEG}")
    
        print(f"Max for MEG is {maxAllMEG}")
    
        # clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar

# %%
allLambdas = np.zeros((len(subjectsToAverage), len(condsToPlot)))
for i in range(len(subjectsToAverage)):
    for j in range(len(condsToPlot)):
        allLambdas[i, j] = np.log10(lambdas[int(condsToPlot[j][3][i])])

allLambdas[allLambdas == -np.inf] = 0

if plotLambdas:

    plt.figure()
    plt.pcolormesh(allLambdas)
    plt.xlabel("Condition")
    plt.ylabel("Subject number")
    cbar = plt.colorbar()
    cbar.set_label(r"$\log_{10}$ optimized lambdas")

# %%
allMaxScores = np.zeros((len(subjectsToAverage), len(condsToPlot)))
for i in range(len(subjectsToAverage)):
    for j in range(len(condsToPlot)):
        allMaxScores[i, j] = condsToPlot[j][4][i]

if plotLambdas:

    plt.figure()
    plt.pcolormesh(allMaxScores)
    plt.xlabel("Condition")
    plt.ylabel("Subject number")
    cbar = plt.colorbar()
    cbar.set_label("Maximum correlation coefficient")

# %%
if optimizeLambda:
    # allMaxScoresAllLambdas = [np.zeros((len(subjectsToAverage) * len(lambdas), len(condsToPlot)))]
    if plotLambdas:
        lambdaXaxisOrig = np.log10(lambdas)
        lambdaXaxisOrig[lambdaXaxisOrig == -np.inf] = 0
        for modality in ["MEG", "EEG"]:
            plt.figure(figsize=[5, 10])
            for i in range(len(condsToPlot)):
                lambdaXaxis = np.tile(lambdaXaxisOrig, (1, len(subjectsToAverage))).squeeze()
                print(lambdaXaxis.shape)
                maxScoresAllLambdas = np.array(condsToPlot[i][5][modality]).flatten()
                # tempLambdas = allLambdas[:, i]
                # tempScores = allMaxScores[:, i]
                comment = condsToPlot[i][0][0].comment
                # result = sp.stats.pearsonr(tempLambdas, tempScores)
                print(f"Lambda X axis has shape {lambdaXaxis.shape}")
                print(f"Max scores all lambdas axis has shape {maxScoresAllLambdas.shape}")
                lambdaXaxis = lambdaXaxis[np.argsort(maxScoresAllLambdas)]
                maxScoresAllLambdas = maxScoresAllLambdas[np.argsort(maxScoresAllLambdas)]
                plt.scatter(lambdaXaxis, maxScoresAllLambdas, color=colors[i], marker='.', facecolors=None, edgecolors=colors[i])
                a, b, c = np.polyfit(lambdaXaxis, maxScoresAllLambdas, deg=2)
                # xseq = np.arange(len(tempLambdas))
                # plt.plot(lambdaXaxis, a*lambdaXaxis**2 + b * lambdaXaxis + c, color=colors[i], label=f"{comment}: r = {np.round(100*result.statistic)/100}, p = {np.round(100*result.pvalue)/100}")
                fitForPlot = a*lambdaXaxisOrig**2 + b * lambdaXaxisOrig + c
                maxIndOfFit = np.argmax(fitForPlot)
                maxOfFit = fitForPlot[maxIndOfFit]
                maxLambdaHere = lambdaXaxisOrig[maxIndOfFit]
                print(f"Best lambda for condition {comment} is 10 to the {maxLambdaHere}, which is lambda index {maxIndOfFit}, with an r of {maxOfFit}")
                plt.plot(lambdaXaxisOrig, fitForPlot, color=colors[i], label=f"{comment}", linewidth=5.5)

            # plt.ylim([.02, .15])
            plt.xlabel(r"$\log_{10}$optimized lambdas")
            plt.ylabel("Maximum correlation coefficient")
            plt.title(modality)
            plt.legend()

# %%
if optimizeLambda:
    trfInd = 3
    sensorInd = evoked.ch_names.index("Fz")
    allTrfs = np.array(condsToPlot[trfInd][6]).mean(axis=0)
    exampleTrf = sp.stats.zscore(allTrfs[:, sensorInd, :].squeeze(), axis=1)

    figLtrf, axLtrf = plt.subplots()

    log10lambdas = np.log10(lambdas)
    if lambdas[0] == 0:
        log10lambdas[0] = 0
    interpForTrf = sp.interpolate.CubicSpline(log10lambdas, exampleTrf)
    newLambdas = np.linspace(log10lambdas[0], log10lambdas[-1], len(lambdas)*4)
    interpedTrf = interpForTrf(newLambdas)

    axLtrf.pcolormesh(condsToPlot[trfInd][0][0].times*1000, newLambdas, interpedTrf)
    axLtrf.invert_yaxis()
    axLtrf.set_xlabel("Time (ms)")

# %%
if plotTopos:
    print(f"Lambda ind {lambdaInd} is {lambdas[lambdaInd]}, max for EEG is {maxAllEEG}, max for MEG is {maxAllMEG}")

# %%
# mne.viz.plot_evoked_topo([evokedValues[i].copy().pick("mag") for i in range(len(evokedValues))])


# %%
# for i in range(len(evokedValues)):
#     evokedValues[i].plot_joint(picks="eeg", title=evokedValues[i].comment)

# %%

# %%
# timeToZoom = .1
# samplesToZoom = int(timeToZoom * evokedAvgQuiet[0].info["sfreq"])
# elecInd = evokedAvgQuiet[0].ch_names.index("Fz")
# plt.figure(figsize=[10,10])
# plt.polar(evokedAvgQuiet[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgQuiet[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="blue", linewidth=3)
# plt.polar(evokedAvgTargetEasy[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetEasy[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="cyan", linewidth=3)
# plt.polar(evokedAvgTargetHard[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetHard[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="green", linewidth=3)
# plt.polar(evokedAvgDistEasy[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetEasy[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="red", linewidth=3)
# plt.polar(evokedAvgDistHard[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetHard[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="brown", linewidth=3)
# plt.plot(evokedAvgQuiet[0].get_data()[25:32,samplesToZoom:-samplesToZoom].T, evokedAvgQuiet[1].get_data()[25:32,samplesToZoom:-samplesToZoom].T)

# %%
subjects_dir = os.path.expandvars("$SUBJECTS_DIR")
if doSources and len(subjectsToAverage) == 1:
    subject = subjectsToAverage[0]
    megLocation = parentDir + subject + subDirs
    initial_time = timesFive[condIndForSources][0]
    evokedHere = condsToPlot[condIndForSources][avgOrAll][complexSubtype].copy().set_eeg_reference("average", projection=True)
    info = evokedHere.info
    noiseCov = eb.load.unpickle(f"{megLocation}highpassNoiseCovariance.pickle")
    # noiseCov = eb.load.unpickle(f"{megLocation}baselineCov_projTrue_fs{fs}.pickle")
    # noiseCov = eb.load.unpickle(f"{megLocation}baselineCov_noProj_fs{fs}.pickle")
    dataCov = eb.load.unpickle(f"{megLocation}dataCovariance_fs{fs}.pickle")
    bem = eb.load.unpickle(f"{subjects_dir}/{subject}/bem/conductorModel.pickle")
    trans = f"{parentDir}{subject}/eegAndMeg/eeg/eeg-trans.fif"

    # pick_ori_MNE = None
    # pick_ori_MNE = "normal"
    pick_ori_MNE = "vector"
    
    # pick_ori_beam = None
    # pick_ori_beam = "normal"
    pick_ori_beam = "vector"
    # pick_ori_beam = "max-power"

    
    # mriVol = subjects_dir + "/" + subject + "/mri/aparc+aseg.mgz"
    # volume_label = None    
    # src = mne.setup_volume_source_space(
    #     subject,
    #     mri=mriVol,
    #     pos=5.0,
    #     bem=bem,
    #     volume_label=volume_label,
    #     subjects_dir=subjects_dir,
    #     add_interpolator=True,  # just for speed, usually this should be True
    #     verbose=True,
    # )
    
    
    mriVol = subjects_dir + "/" + subject + "/mri/aseg.mgz"
    volume_label = [
        "Left-Thalamus-Proper",
        "Right-Thalamus-Proper",
        "Brain-Stem",
    ]    
    src_surf = mne.setup_source_space(
        subject, spacing="oct6", add_dist=False, subjects_dir=subjects_dir
    )    
    src_vol = mne.setup_volume_source_space(
        subject,
        mri=mriVol,
        pos=5.0,
        bem=bem,
        volume_label=volume_label,
        subjects_dir=subjects_dir,
        add_interpolator=True,  # just for speed, usually this should be True
        verbose=True,
    )
    src = src_surf + src_vol



    src.plot(subjects_dir=subjects_dir)
    
    fwd = mne.make_forward_solution(
        info,
        trans,
        src,
        bem,
        mindist=5.0,  # ignore sources<=5mm from innerskull
        meg=True,
        eeg=True,
        n_jobs=None,
    )

# %%
# if doSources and len(subjectsToAverage) == 1:
#     from mpl_toolkits.mplot3d import Axes3D
#     dip = []
#     n_dipoles = 2
#     # figDip, axDip = plt.subplots(n_dipoles, 4, figsize=(25, 15), subplot_kw=dict(projection='3d'))
#     initial_time = timesFive[condIndForSources][0]
#     for timeInd in range(4):
#         evokedForSources = evokedHere.copy().crop(tmin=timesFive[condIndForSources][timeInd]-.0005, tmax=timesFive[condIndForSources][timeInd]+.0005)

#         # dip1, res1 = mne.fit_dipole(evokedForSources, 
#         #                           noiseCov,
#         #                           bem, 
#         #                           trans=trans)
        
#         # dip2, res2 = mne.fit_dipole(res1, 
#         #                           noiseCov,
#         #                           bem, 
#         #                           trans=trans)

        
#         dip.append(mne.beamformer.trap_music(evokedForSources, 
#                                          fwd, 
#                                          noiseCov, 
#                                          n_dipoles=n_dipoles,
#                                          return_residual=False,
#                                         )
#                   )
#         for i in range(n_dipoles):
#             # dip[timeInd][i].plot_locations(trans, subject, subjects_dir, mode="orthoview", ax=axDip[i, timeInd])
#             dip[timeInd][i].plot_locations(trans, subject, subjects_dir, mode="orthoview")
#         # dip1[0].plot_locations(trans, subject, subjects_dir, mode="orthoview")
#         # dip2.plot_locations(trans, subject, subjects_dir, mode="orthoview")

# %%
if doSources and len(subjectsToAverage) == 1:
    # snr = 3.0  # use smaller SNR for raw data
    snr = 9
    inv_method = "dSPM"  # sLORETA, MNE, dSPM
    # loose = dict(volume=1.0)
    loose = "auto"
    depth = 0.8
    # depth = 10.0
    lambda2 = 1.0 / snr**2

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info, fwd, noiseCov, depth=depth, loose=loose, verbose=True
    )

    stc = mne.minimum_norm.apply_inverse(
        evokedHere, inverse_operator, lambda2, inv_method, pick_ori=pick_ori_MNE
    )

    brain = stc.plot(
        hemi="both",
        src=inverse_operator["src"],
        views="coronal",
        initial_time=initial_time,
        subjects_dir=subjects_dir,
        brain_kwargs=dict(silhouette=True),
        smoothing_steps=7,
        show_traces=True,
    )
    mne.viz.set_3d_title(brain, f"MNE: {stc.subject}", size=20)

    # bem_fs = eb.load.unpickle(f"{subjects_dir}/fsaverage/bem/conductorModel.pickle")
    # src_fs = mne.setup_volume_source_space(
    #     "fsaverage",
    #     mri=subjects_dir + "/fsaverage/mri/aparc+aseg.mgz",
    #     pos=5.0,
    #     bem=bem_fs,
    #     volume_label=volume_label,
    #     subjects_dir=subjects_dir,
    #     add_interpolator=True,  # just for speed, usually this should be True
    #     verbose=True,
    # )
    # morph = mne.compute_source_morph(
    #     src,
    #     subject_from=subject,
    #     subject_to="fsaverage",
    #     src_to=src_fs,
    #     subjects_dir=subjects_dir,
    #     niter_sdr=[5, 5, 2],
    #     niter_affine=[5, 5, 2],
    #     zooms=7,  # just for speed
    #     verbose=True,
    # )
    # stc_fs = morph.apply(stc)

    # brain = stc_fs.plot_3d(
    #     hemi="both",
    #     src=src_fs,
    #     views="coronal",
    #     initial_time=initial_time,
    #     subjects_dir=subjects_dir,
    #     brain_kwargs=dict(silhouette=True),
    #     smoothing_steps=0,
    #     show_traces=True,
    # )

# %%
if doSources:
    # weight_norm = None
    weight_norm = "unit-noise-gain-invariant"
    # weight_norm = "nai"
    depthBeam = 0.8
    reg = 0.05
    lcmv = mne.beamformer.make_lcmv(info=info, 
                                    forward=fwd, 
                                    data_cov=dataCov, 
                                    reg=reg, 
                                    noise_cov=noiseCov, 
                                    label=None, 
                                    pick_ori=pick_ori_beam, 
                                    rank="info", 
                                    weight_norm=weight_norm, 
                                    reduce_rank=False, 
                                    depth=depthBeam, 
                                    inversion="matrix", 
                                    verbose=None)
    stcBeam = mne.beamformer.apply_lcmv(evokedHere, lcmv, verbose="info")

    brain2 = stcBeam.plot(
    hemi="both",
    src=inverse_operator["src"],
    views="coronal",
    initial_time=initial_time,
    subjects_dir=subjects_dir,
    brain_kwargs=dict(silhouette=True),
    smoothing_steps=7,
    show_traces=True,
    )
    mne.viz.set_3d_title(brain2, f"Beamformer: {stc.subject}", size=20)
        
        

# %%
# if doSources:
    # alpha = 0.05
    # # alpha = 0.00000005
    # loose = "auto"
    # depth = 0.8
    # stcGamma = mne.inverse_sparse.gamma_map(
    #     evokedHere,
    #     fwd,
    #     noiseCov,
    #     alpha,
    #     loose=loose,
    #     depth=depth,
    #     tol=1e-4,
    # )
    # brain3 = stcGamma.plot_3d(
    # hemi="both",
    # src=inverse_operator["src"],
    # views="coronal",
    # initial_time=initial_time,
    # subjects_dir=subjects_dir,
    # brain_kwargs=dict(silhouette=True),
    # smoothing_steps=0,
    # show_traces=True,
    # )


# %%
def doWarp(origT, test, warpCenter, lenWarp, ampWarp, fs):
    warper = sp.interpolate.CubicSpline(origT, test)
    warpCenterInd = np.abs(origT - warpCenter).argmin()
    tWarp = np.linspace(0, 1, int(fs * lenWarp))
    warpSin = ampWarp * np.sin(2 * np.pi * tWarp)
    warpAll = np.zeros(origT.shape)
    warpStartInd = warpCenterInd - int(lenWarp * fs / 2)
    warpAll[warpStartInd: warpStartInd + int(lenWarp * fs)] += warpSin
    newTderiv = origTderiv + warpAll
    newT = np.cumsum(newTderiv) + evokedHere.times[0]
    newTest = warper(newT)
    print(f"Warp center is at {warpCenter}")
    return newTest, warpSin, newT


if doWarpTesting:

    evokedForWarpTest = condsToPlot[condIndForSources][avgOrAll].copy()
    evokedNew = []
    for i in range(len(complexSubtypeNames)):
        test = evokedForWarpTest[i].data.T
        lenSig = len(evokedForWarpTest[i].times)
        origTderiv = np.ones(lenSig) / fs
        origT = np.cumsum(origTderiv) + evokedForWarpTest[i].times[0]
        warpCenter = timesFive[condIndForSources][0]
    
        lenWarp = .0035
        ampWarp = -.0003
        # plt.figure()
        # plt.plot(origT)
        # plt.plot(newT)
        
        newTest, warpSin, newT = doWarp(origT, test, warpCenter, lenWarp, ampWarp, fs)
        evokedNew.append(mne.EvokedArray(data=newTest.T, info=evokedForWarpTest[i].info, tmin=evokedForWarpTest[i].times[0]))
    # lenWarp = .0035
    # ampWarp /= 2
    # newTest = doWarp(origT, newTest, warpCenter, lenWarp, ampWarp, fs)
    # newTest = doWarp(origT, newTest, warpCenter, lenWarp, ampWarp, fs)
    
    # plt.figure()
    # plt.plot(origT, test, color="blue", alpha=.5)
    # plt.plot(origT, newTest, color="red", alpha=.5)
    # plt.legend(["Original", "Warped"])
    
    
    figJointEEGnew = evokedForWarpTest[0].plot_joint(picks="eeg", times=timesFive[condIndForSources], title=evokedForWarpTest[0].comment, ts_args={"gfp": True})
    
    if custLines:
        for line in figJointEEGnew.axes[len(timesFive[condIndForSources])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
            
    if not eegOnly:
            
        figJointMEGnew = evokedForWarpTest[0].plot_joint(picks="meg", times=timesFive[condIndForSources], title=evokedForWarpTest[0].comment, ts_args={"gfp": True})
    
        if custLines:
            for line in figJointMEGnew.axes[len(timesFive[condIndForSources])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
                line.set_linewidth(jointLinewidth)
                line.set_alpha(jointAlpha)
    
                    
    condsToPlotNew = condsToPlot.copy()
    for i in range(len(condsToPlotNew)):
        condsToPlotNew[i] = list(condsToPlotNew[i])
        for j in range(len(condsToPlotNew[i])):
            condsToPlotNew[i][j] = list(condsToPlotNew[i][j])
    condsToPlotNew[condIndForSources][avgOrAll] = evokedNew
    timesFiveNew = createPeakTimes(avgOrAll, condsToPlotNew, peakRanges, numPeaksPerRange, useEEGonly, useMEGonly, order, plotTimes, eegOnly)
    print(f"New center is at {timesFiveNew[condIndForSources][0]}")
    
    
    figJointEEGnew2 = evokedNew[0].plot_joint(picks="eeg", times=timesFiveNew[condIndForSources], title=evokedNew[0].comment, ts_args={"gfp": True})
    
    if custLines:
        for line in figJointEEGnew2.axes[len(timesFiveNew[condIndForSources])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
            
    if not eegOnly:
            
        figJointMEGnew2 = evokedNew[0].plot_joint(picks="meg", times=timesFiveNew[condIndForSources], title=evokedNew[0].comment, ts_args={"gfp": True})
    
        if custLines:
            for line in figJointMEGnew2.axes[len(timesFive[condIndForSources])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
                line.set_linewidth(jointLinewidth)
                line.set_alpha(jointAlpha)

# %%
if doPCplot:

    figSaveDir = "/Users/karl/Downloads/testPngs/sensorPlotFrames/"
    
    figSize = [7, 4]
    linewidthPCplot = 2
    vertLineColor = "silver"
    
    MEGpcSpeech = calcPC(evokedAvgMixAll[0][0].data[np.newaxis, :157, :], True, None).squeeze()
    EEGpcSpeech = calcPC(evokedAvgMixAll[0][0].data[np.newaxis, 157:, :], True, None).squeeze()
    MEGpcClicks = calcPC(evokedAvgClicksComplete[0][0].data[np.newaxis, :157, :], True, None).squeeze()
    EEGpcClicks = calcPC(evokedAvgClicksComplete[0][0].data[np.newaxis, 157:, :], True, None).squeeze()
    t = evokedAvgMixAll[0][0].times * 1000


    figSpeechEEG, axSpeechEEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figSpeechMEG, axSpeechMEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figClicksEEG, axClicksEEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figClicksMEG, axClicksMEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)

    figSpeechTitle, axSpeechTitle = plt.subplots(figsize=[12,1])
    figClicksTitle, axClicksTitle = plt.subplots(figsize=[12,1])

    figSpeechTime, axSpeechTime = plt.subplots(figsize=[5,1])
    figClicksTime, axClicksTime = plt.subplots(figsize=[5,1])
    
    figSpeechPeak, axSpeechPeak = plt.subplots(figsize=[5,1])
    figClicksPeak, axClicksPeak = plt.subplots(figsize=[5,1])

    dpi = 175
    
    for i in range(len(t)):
    # for i in range(40, 41):
    # for i in range(100, 101):
    
        lineTime = t[i]

        axSpeechEEG.cla()
        axSpeechEEG.axhline(y=0, color="white", linewidth=.6)
        axSpeechEEG.plot(t, EEGpcSpeech, color="white", linewidth=linewidthPCplot)
        axSpeechEEG.set_title("EEG Sensor PC", color="white", fontsize=28)
        axSpeechEEG.set_xlabel("Time (ms)", color="white")
        axSpeechEEG.tick_params(colors="white")
        for spine in axSpeechEEG.spines.values():
            spine.set_visible(False)
        axSpeechEEG.set_facecolor("black")
        axSpeechEEG.set_yticks([])
        axSpeechEEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axSpeechEEG.text(.01, .92, "Wave V", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.09, .03, "SN10", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.3, .12, "Na", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.39, .86, "Pa", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.475, .07, "Nb", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.63, .63, "P50", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        figSpeechEEG.set_facecolor("black")
        figSpeechEEG.savefig(f"{figSaveDir}speechEEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        axSpeechMEG.cla()
        axSpeechMEG.axhline(y=0, color="white", linewidth=.6)
        axSpeechMEG.plot(t, MEGpcSpeech, color="white", linewidth=linewidthPCplot)
        axSpeechMEG.set_title("MEG Sensor PC", color="white", fontsize=28)
        axSpeechMEG.set_xlabel("Time (ms)", color="white")
        axSpeechMEG.tick_params(colors="white")
        for spine in axSpeechMEG.spines.values():
            spine.set_visible(False)
        axSpeechMEG.set_facecolor("black")
        axSpeechMEG.set_yticks([])
        axSpeechMEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axSpeechMEG.text(.01, .5, "Wave V", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.09, .1, "SN10", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.3, .01, "Na", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.35, .94, "Pa", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.5, .015, "Nb", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.685, .75, "P50", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        figSpeechMEG.set_facecolor("black")
        figSpeechMEG.savefig(f"{figSaveDir}speechMEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        axClicksEEG.cla()
        axClicksEEG.axhline(y=0, color="white", linewidth=.6)
        axClicksEEG.plot(t, EEGpcClicks, color="white", linewidth=linewidthPCplot)
        axClicksEEG.set_title("EEG Sensor PC", color="white", fontsize=28)
        axClicksEEG.set_xlabel("Time (ms)", color="white")
        axClicksEEG.tick_params(colors="white")
        for spine in axClicksEEG.spines.values():
            spine.set_visible(False)
        axClicksEEG.set_facecolor("black")
        axClicksEEG.set_yticks([])
        axClicksEEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axClicksEEG.text(.01, .92, "Wave V", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.15, .06, "SN10", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.32, .035, "Na", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.39, .8, "Pa", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.585, .13, "Nb", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.72, .55, "P50", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        figClicksEEG.set_facecolor("black")
        figClicksEEG.savefig(f"{figSaveDir}clicksEEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        axClicksMEG.cla()
        axClicksMEG.axhline(y=0, color="white", linewidth=.6)
        axClicksMEG.plot(t, MEGpcClicks, color="white", linewidth=linewidthPCplot)
        axClicksMEG.set_title("MEG Sensor PC", color="white", fontsize=28)
        axClicksMEG.set_xlabel("Time (ms)", color="white")
        axClicksMEG.tick_params(colors="white")
        for spine in axClicksMEG.spines.values():
            spine.set_visible(False)
        axClicksMEG.set_facecolor("black")
        axClicksMEG.set_yticks([])
        axClicksMEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axClicksMEG.text(.035, .73, "Wave V", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.13, .43, "SN10", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.29, .235, "Na", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.35, .94, "Pa", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.55, .026, "Nb", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.75, .93, "P50", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        figClicksMEG.set_facecolor("black")
        figClicksMEG.savefig(f"{figSaveDir}clicksMEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        axSpeechTitle.cla()
        axSpeechTitle.text(.001, .2, f"Speech TRF", color="white", fontsize=60, transform=axSpeechTitle.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axSpeechTitle.set_facecolor("black")
        figSpeechTitle.set_facecolor("black")
        figSpeechTitle.savefig(f"{figSaveDir}speechTitle{i:04d}.png", dpi=dpi)
        
        axClicksTitle.cla()
        axClicksTitle.text(.001, .2, f"Click TRF", color="white", fontsize=60, transform=axSpeechTitle.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axClicksTitle.set_facecolor("black")
        figClicksTitle.set_facecolor("black")
        figClicksTitle.savefig(f"{figSaveDir}clicksTitle{i:04d}.png", dpi=dpi)

        axSpeechTime.cla()
        axSpeechTime.text(.001, .2, f"t = {int(np.round(lineTime))} ms", color="white", fontsize=60, transform=axSpeechTime.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axSpeechTime.set_facecolor("black")
        figSpeechTime.set_facecolor("black")
        figSpeechTime.savefig(f"{figSaveDir}speechTime{i:04d}.png", dpi=dpi)
        
        axClicksTime.cla()
        axClicksTime.text(.001, .2, f"t = {int(np.round(lineTime))} ms", color="white", fontsize=60, transform=axSpeechTime.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axClicksTime.set_facecolor("black")
        figClicksTime.set_facecolor("black")
        figClicksTime.savefig(f"{figSaveDir}clicksTime{i:04d}.png", dpi=dpi)

        speechPeakText = "        "
        clicksPeakText = "        "
        if 7 < lineTime < 10:
            speechPeakText = "Wave V"
            clicksPeakText = "Wave V"
        elif 10 < lineTime < 13:
            speechPeakText = "SN10"
            clicksPeakText = "SN10"
        elif 18 < lineTime < 22:
            speechPeakText = "Na"
            clicksPeakText = "Na"
        elif 24 < lineTime < 29:
            speechPeakText = "Pa"
            clicksPeakText = "Pa"
        elif 32 < lineTime < 36:
            speechPeakText = "Nb"
        elif 40 < lineTime < 45:
            clicksPeakText = "Nb"
        elif 48 < lineTime < 54:
            speechPeakText = "P50"
        elif 51 < lineTime < 56:
            clicksPeakText = "P50"
            
        
        axSpeechPeak.cla()
        axSpeechPeak.text(.001, .2, speechPeakText, color="white", fontsize=70, transform=axSpeechPeak.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axSpeechPeak.set_facecolor("black")
        figSpeechPeak.set_facecolor("black")
        figSpeechPeak.savefig(f"{figSaveDir}speechPeak{i:04d}.png", dpi=dpi)
        
        axClicksPeak.cla()
        axClicksPeak.text(.001, .2, clicksPeakText, color="white", fontsize=70, transform=axClicksPeak.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axClicksPeak.set_facecolor("black")
        figClicksPeak.set_facecolor("black")
        figClicksPeak.savefig(f"{figSaveDir}clicksPeak{i:04d}.png", dpi=dpi)

# %%
if doPCplot:

    figSaveDir = "/Users/karl/Downloads/testPngs/sensorPlotFrames/"
    
    figSize = [7, 4]
    linewidthPCplot = 2
    vertLineColor = "silver"
    savePCfigs = False
    
    MEGpcTarget = calcPC(evokedAvgTargetAll[0][0].data[np.newaxis, :157, :], True, None).squeeze()
    EEGpcTarget = calcPC(evokedAvgTargetAll[0][0].data[np.newaxis, 157:, :], True, None).squeeze()
    MEGpcDist   = calcPC(evokedAvgDistAll[0][0].data[np.newaxis, :157, :], True, MEGpcTarget).squeeze()
    EEGpcDist   = calcPC(evokedAvgDistAll[0][0].data[np.newaxis, 157:, :], True, None).squeeze()
    MEGpcSpeech = calcPC(evokedAvgMixAll[0][0].data[np.newaxis, :157, :], True, None).squeeze()
    EEGpcSpeech = calcPC(evokedAvgMixAll[0][0].data[np.newaxis, 157:, :], True, None).squeeze()
    MEGpcClicks = calcPC(evokedAvgClicksComplete[0][0].data[np.newaxis, :157, :], True, None).squeeze()
    EEGpcClicks = calcPC(evokedAvgClicksComplete[0][0].data[np.newaxis, 157:, :], True, None).squeeze()
    t = evokedAvgMixAll[0][0].times * 1000


    figSpeechEEG, axSpeechEEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figSpeechMEG, axSpeechMEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figSpeechAllEEG, axSpeechAllEEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figSpeechAllMEG, axSpeechAllMEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figClicksEEG, axClicksEEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)
    figClicksMEG, axClicksMEG = plt.subplots(nrows=1, ncols=1, figsize=figSize)

    figSpeechTitle, axSpeechTitle = plt.subplots(figsize=[12,1])
    figClicksTitle, axClicksTitle = plt.subplots(figsize=[12,1])

    dpi = 175
    
    # for i in range(len(t)):
    for i in range(40, 41):
    # for i in range(100, 101):
    
        lineTime = t[i]

        axSpeechEEG.cla()
        axSpeechEEG.axhline(y=0, color="white", linewidth=.6)
        axSpeechEEG.plot(t, EEGpcSpeech, color="white", linewidth=linewidthPCplot)
        axSpeechEEG.set_title("EEG Sensor PC", color="white", fontsize=28)
        axSpeechEEG.set_xlabel("Time (ms)", color="white")
        axSpeechEEG.tick_params(colors="white")
        for spine in axSpeechEEG.spines.values():
            spine.set_visible(False)
        axSpeechEEG.set_facecolor("black")
        axSpeechEEG.set_yticks([])
        axSpeechEEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axSpeechEEG.text(.01, .92, "Wave V", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.09, .03, "SN10", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.3, .12, "Na", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.39, .86, "Pa", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.475, .07, "Nb", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechEEG.text(.63, .63, "P50", transform=axSpeechEEG.transAxes, color=vertLineColor, fontsize=14)
        figSpeechEEG.set_facecolor("black")
        if savePCfigs:
            figSpeechEEG.savefig(f"{figSaveDir}speechEEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        axSpeechMEG.cla()
        axSpeechMEG.axhline(y=0, color="white", linewidth=.6)
        axSpeechMEG.plot(t, MEGpcSpeech, color="white", linewidth=linewidthPCplot)
        axSpeechMEG.set_title("MEG Sensor PC", color="white", fontsize=28)
        axSpeechMEG.set_xlabel("Time (ms)", color="white")
        axSpeechMEG.tick_params(colors="white")
        for spine in axSpeechMEG.spines.values():
            spine.set_visible(False)
        axSpeechMEG.set_facecolor("black")
        axSpeechMEG.set_yticks([])
        axSpeechMEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axSpeechMEG.text(.01, .5, "Wave V", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.09, .1, "SN10", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.3, .01, "Na", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.35, .94, "Pa", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.5, .015, "Nb", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        axSpeechMEG.text(.685, .75, "P50", transform=axSpeechMEG.transAxes, color=vertLineColor, fontsize=14)
        figSpeechMEG.set_facecolor("black")
        if savePCfigs:
            figSpeechMEG.savefig(f"{figSaveDir}speechMEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        
        axSpeechAllEEG.cla()
        axSpeechAllEEG.axhline(y=0, color="white", linewidth=.6)
        axSpeechAllEEG.plot(t, EEGpcSpeech, color="blue", linewidth=linewidthPCplot)
        axSpeechAllEEG.plot(t, EEGpcTarget, color="green", linewidth=linewidthPCplot)
        axSpeechAllEEG.plot(t, EEGpcDist, color="red", linewidth=linewidthPCplot)
        axSpeechAllEEG.set_title("EEG Sensor PC", color="white", fontsize=28)
        axSpeechAllEEG.set_xlabel("Time (ms)", color="white")
        axSpeechAllEEG.tick_params(colors="white")
        for spine in axSpeechAllEEG.spines.values():
            spine.set_visible(False)
        axSpeechAllEEG.set_facecolor("black")
        axSpeechAllEEG.set_yticks([])
        axSpeechAllEEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        # axSpeechAllEEG.text(.01, .92, "Wave V", transform=axSpeechAllEEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllEEG.text(.09, .03, "SN10", transform=axSpeechAllEEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllEEG.text(.3, .12, "Na", transform=axSpeechAllEEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllEEG.text(.39, .86, "Pa", transform=axSpeechAllEEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllEEG.text(.475, .07, "Nb", transform=axSpeechAllEEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllEEG.text(.63, .63, "P50", transform=axSpeechAllEEG.transAxes, color=vertLineColor, fontsize=14)
        figSpeechAllEEG.set_facecolor("black")
        if savePCfigs:
            figSpeechAllEEG.savefig(f"{figSaveDir}speechEEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        axSpeechAllMEG.cla()
        axSpeechAllMEG.axhline(y=0, color="white", linewidth=.6)
        axSpeechAllMEG.plot(t, MEGpcSpeech, color="blue", linewidth=linewidthPCplot)
        axSpeechAllMEG.plot(t, MEGpcTarget, color="green", linewidth=linewidthPCplot)
        axSpeechAllMEG.plot(t, MEGpcDist, color="red", linewidth=linewidthPCplot)
        axSpeechAllMEG.set_title("MEG Sensor PC", color="white", fontsize=28)
        axSpeechAllMEG.set_xlabel("Time (ms)", color="white")
        axSpeechAllMEG.tick_params(colors="white")
        for spine in axSpeechAllMEG.spines.values():
            spine.set_visible(False)
        axSpeechAllMEG.set_facecolor("black")
        axSpeechAllMEG.set_yticks([])
        axSpeechAllMEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        # axSpeechAllMEG.text(.01, .5, "Wave V", transform=axSpeechAllMEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllMEG.text(.09, .1, "SN10", transform=axSpeechAllMEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllMEG.text(.3, .01, "Na", transform=axSpeechAllMEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllMEG.text(.35, .94, "Pa", transform=axSpeechAllMEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllMEG.text(.5, .015, "Nb", transform=axSpeechAllMEG.transAxes, color=vertLineColor, fontsize=14)
        # axSpeechAllMEG.text(.685, .75, "P50", transform=axSpeechAllMEG.transAxes, color=vertLineColor, fontsize=14)
        figSpeechAllMEG.set_facecolor("black")
        if savePCfigs:
            figSpeechAllMEG.savefig(f"{figSaveDir}speechMEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        
        axClicksEEG.cla()
        axClicksEEG.axhline(y=0, color="white", linewidth=.6)
        axClicksEEG.plot(t, EEGpcClicks, color="white", linewidth=linewidthPCplot)
        axClicksEEG.set_title("EEG Sensor PC", color="white", fontsize=28)
        axClicksEEG.set_xlabel("Time (ms)", color="white")
        axClicksEEG.tick_params(colors="white")
        for spine in axClicksEEG.spines.values():
            spine.set_visible(False)
        axClicksEEG.set_facecolor("black")
        axClicksEEG.set_yticks([])
        axClicksEEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axClicksEEG.text(.01, .92, "Wave V", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.15, .06, "SN10", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.32, .035, "Na", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.39, .8, "Pa", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.585, .13, "Nb", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksEEG.text(.72, .55, "P50", transform=axClicksEEG.transAxes, color=vertLineColor, fontsize=14)
        figClicksEEG.set_facecolor("black")
        if savePCfigs:
            figClicksEEG.savefig(f"{figSaveDir}clicksEEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        
        axClicksMEG.cla()
        axClicksMEG.axhline(y=0, color="white", linewidth=.6)
        axClicksMEG.plot(t, MEGpcClicks, color="white", linewidth=linewidthPCplot)
        axClicksMEG.set_title("MEG Sensor PC", color="white", fontsize=28)
        axClicksMEG.set_xlabel("Time (ms)", color="white")
        axClicksMEG.tick_params(colors="white")
        for spine in axClicksMEG.spines.values():
            spine.set_visible(False)
        axClicksMEG.set_facecolor("black")
        axClicksMEG.set_yticks([])
        axClicksMEG.axvline(x=lineTime, color=vertLineColor, alpha=0.75, linewidth=linewidthPCplot)
        axClicksMEG.text(.035, .73, "Wave V", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.13, .43, "SN10", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.29, .235, "Na", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.35, .94, "Pa", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.55, .026, "Nb", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        axClicksMEG.text(.75, .93, "P50", transform=axClicksMEG.transAxes, color=vertLineColor, fontsize=14)
        figClicksMEG.set_facecolor("black")
        if savePCfigs:
            figClicksMEG.savefig(f"{figSaveDir}clicksMEG{i:04d}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        axSpeechTitle.cla()
        axSpeechTitle.text(.001, .2, f"Speech TRF: t = {int(np.round(lineTime))} ms", color="white", fontsize=60, transform=axSpeechTitle.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axSpeechTitle.set_facecolor("black")
        figSpeechTitle.set_facecolor("black")
        if savePCfigs:
            figSpeechTitle.savefig(f"{figSaveDir}speechTitle{i:04d}.png", dpi=dpi)
        
        axClicksTitle.cla()
        axClicksTitle.text(.001, .2, f"Click TRF: t = {int(np.round(lineTime))} ms", color="white", fontsize=60, transform=axSpeechTitle.transAxes, horizontalalignment="left", verticalalignment="baseline")
        axClicksTitle.set_facecolor("black")
        figClicksTitle.set_facecolor("black")
        if savePCfigs:
            figClicksTitle.savefig(f"{figSaveDir}clicksTitle{i:04d}.png", dpi=dpi)

# %% jp-MarkdownHeadingCollapsed=true
if doClusters:
    
    condInd1 = 3
    condInd2 = 4
    
    # n_permutations = 1024
    n_permutations = "all"
    
    # threshold = dict(start=4, step=4)
    n_observations = 12
    pval = 0.05  # arbitrary
    df = n_observations - 1  # degrees of freedom for the test
    threshold = sp.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    speechEEGMat = []
    clicksEEGMat = []
    speechMEGMat = []
    clicksMEGMat = []
    
    for subjEvoked in condsToPlot[condInd1][1][0]:
        speechEEGMat.append(subjEvoked.copy().get_data(picks="eeg").T)
        speechMEGMat.append(subjEvoked.copy().get_data(picks="meg").T)
    for subjEvoked in condsToPlot[condInd2][1][0]:
        clicksEEGMat.append(subjEvoked.copy().get_data(picks="eeg").T)
        clicksMEGMat.append(subjEvoked.copy().get_data(picks="meg").T)
    
    speechEEGMat = np.array(speechEEGMat)
    speechMEGMat = np.array(speechMEGMat)
    clicksEEGMat = np.array(clicksEEGMat)
    clicksMEGMat = np.array(clicksMEGMat)
    
    speechEEGnewEvoked = mne.EvokedArray(data=np.nanmean(speechEEGMat, axis=0).T, info=condsToPlot[condInd1][0][0].copy().pick("eeg").info, 
                                         tmin=condsToPlot[condInd1][0][0].copy().pick("eeg").times[0], comment=condsToPlot[condInd1][0][0].copy().pick("eeg").comment)
    speechMEGnewEvoked = mne.EvokedArray(data=np.nanmean(speechMEGMat, axis=0).T, info=condsToPlot[condInd1][0][0].copy().pick("meg").info, 
                                         tmin=condsToPlot[condInd1][0][0].copy().pick("meg").times[0], comment=condsToPlot[condInd1][0][0].copy().pick("meg").comment)
    clicksEEGnewEvoked = mne.EvokedArray(data=np.nanmean(clicksEEGMat, axis=0).T, info=condsToPlot[condInd2][0][0].copy().pick("eeg").info, 
                                         tmin=condsToPlot[condInd2][0][0].copy().pick("eeg").times[0], comment=condsToPlot[condInd2][0][0].copy().pick("eeg").comment)
    clicksMEGnewEvoked = mne.EvokedArray(data=np.nanmean(clicksMEGMat, axis=0).T, info=condsToPlot[condInd2][0][0].copy().pick("meg").info, 
                                         tmin=condsToPlot[condInd2][0][0].copy().pick("meg").times[0], comment=condsToPlot[condInd2][0][0].copy().pick("meg").comment)
    
    # This will just use the last subject for sensor adjacency matrices, but they should be very similar across people and runs etc.
    eegAdj = mne.channels.find_ch_adjacency(subjEvoked.copy().pick("eeg").info, ch_type="eeg")
    megAdj = mne.channels.find_ch_adjacency(subjEvoked.copy().pick("meg").info, ch_type="mag")
    
    t_obs_eeg, clusters_eeg, p_values_eeg, H0_eeg = mne.stats.spatio_temporal_cluster_1samp_test(speechEEGMat - clicksEEGMat, adjacency=eegAdj[0], 
                                                                                                 out_type="mask", threshold=threshold, n_permutations=n_permutations)
    t_obs_meg, clusters_meg, p_values_meg, H0_meg = mne.stats.spatio_temporal_cluster_1samp_test(speechMEGMat - clicksMEGMat, adjacency=megAdj[0], 
                                                                                                 out_type="mask", threshold=threshold, n_permutations=n_permutations)

# %% jp-MarkdownHeadingCollapsed=true
if doClusters:

    # Do EEG first. This is adapted from MNE examples
    
    p_accept = 0.05
    
    sigClusterInds_eeg = np.where(p_values_eeg < p_accept)[0]
    
    # Customize this here if you already know what it's going to be, to only look at certain cluster(s)
    # sigClusterInds_eeg = sigClusterInds_eeg[1:3]
    
    significant_points = clusters_eeg[sigClusterInds_eeg[0]]
    for i in sigClusterInds_eeg[1:]:
        significant_points = significant_points | clusters_eeg[i]
    
    
    # significant_points = p_values_eeg.reshape(t_obs_eeg.shape).T < 0.05
    significant_points = significant_points.T
    # We need an evoked object to plot the image to be masked
    evokedDiffEEG = mne.combine_evoked(
        [speechEEGnewEvoked, clicksEEGnewEvoked], weights=[1, -1]
    )  # calculate difference wave
    time_unit = "ms"
    time_unit_dict = dict(time_unit=time_unit)
    
    timesHereMeans = []
    for i, clusterInd in enumerate(sigClusterInds_eeg):
        timesHere = np.where(clusters_eeg[clusterInd])[0] / 5000
        timesHereMeans.append(np.mean(timesHere))
    
    diffJointEEG = evokedDiffEEG.plot_joint(
        title="Speech vs. click", ts_args=time_unit_dict, topomap_args=time_unit_dict, times=np.sort(timesHereMeans), 
    )  # show difference wave
    if custLines:
        for line in diffJointEEG.axes[3].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
    # Create ROIs by checking channel labels
    selections = mne.channels.make_1020_channel_selections(evokedDiffEEG.info, midline="12z")
    
    # Visualize the results
    fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
    axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
    evokedDiffEEG.plot_image(
        axes=axes,
        group_by=selections,
        colorbar=False,
        show=False,
        mask=significant_points,
        mask_alpha=1,
        show_names="all",
        titles=None,
        **time_unit_dict,
    )
    plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="Z-scored potential")
    
    plt.show()
    
    for i, clusterInd in enumerate(sigClusterInds_eeg):
        timesHere = np.where(clusters_eeg[clusterInd])[0] / 5000
        chansHere = np.where(clusters_eeg[clusterInd])[1]
        print(f"EEG cluster {i+1}: Times {timesHere[0]} to {timesHere[-1]}; p = {p_values_eeg[clusterInd]}")
        print(f"EEG cluster {i+1}: Number of channels is {len(np.unique(chansHere))}")

# %%
if doClusters:

    # We subselect clusters that we consider significant at an arbitrarily
    # picked alpha level: "p_accept".
    # NOTE: remember the caveats with respect to "significant" clusters that
    # we mentioned in the introduction of this tutorial!
    
    # This cell is copied and pasted from below, working for the MEG, to here to do the same for the EEG
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # sigClusterInds_meg = np.where(p_values_meg < p_accept)[0]
    
    # configure variables for visualization
    colors = {"M: All (real)": "crimson", "T: Clicks complete (real)": "steelblue"}
    linestyles = {"M: All (real)": "-", "T: Clicks complete (real)": "--"}
    
    # organize data for plotting
    # evokeds = {cond: epochs[cond].average() for cond in event_id}
    evokedsEeg = [speechEEGnewEvoked, clicksEEGnewEvoked]
    
    
    # evokedDiffMEG = mne.combine_evoked(
    #     [speechMEGnewEvoked, clicksMEGnewEvoked], weights=[1, -1]
    # )  # calculate difference wave
    
    layout = mne.channels.find_layout(speechEEGnewEvoked.info, ch_type="eeg")
    
    # Create ROIs by checking channel labels
    # selections = mne.channels.make_1020_channel_selections(evokedDiffEEG.info, midline="12z")
    # leftSensors = layout.pos[:, 0] < 0.45
    # rightSensors = layout.pos[:, 0] >= 0.45
    # lowerSensors = layout.pos[:, 1] < 0.55
    # upperSensors = layout.pos[:, 1] >= 0.55
    # selections_meg = {"Upper Left": np.where(leftSensors & upperSensors)[0],
    #                   "Lower Left": np.where(leftSensors & lowerSensors)[0],
    #                   "Upper Right": np.where(rightSensors & upperSensors)[0],
    #                   "Lower Right": np.where(rightSensors & lowerSensors)[0],
    #                  }
    # significant_points_meg = clusters_meg[sigClusterInds_meg[0]]
    # for i in sigClusterInds_meg[1:]:
    #     significant_points_meg = significant_points_meg | clusters_meg[i]
    
    # significant_points_meg = significant_points_meg.T
    
    # Visualize the results
    # figMegImage, axesMegImage = plt.subplots(nrows=4, figsize=(8, 8))
    topoNames = ["topo1", "topo2", "topo3"]
    
    # mosaicGridEeg = [[".", topoNames[0], topoNames[1], topoNames[2], "."],
    #                  [".", "butterfly", "butterfly", "butterfly", "."],
    #                  ["channelGroup1", "Left", "Left", "Left", "."],
    #                  ["channelGroup2", "Midline", "Midline", "Midline", "colorbar"],
    #                  ["channelGroup3", "Right", "Right", "Right", "."],
    #                 ]
    
    mosaicGridEeg = [[".", topoNames[0], topoNames[1], topoNames[2], "."],
                     [".", "butterfly", "butterfly", "butterfly", "."],
                     ["channelGroup1", "Image", "Image", "Image", "."],
                     ["channelGroup2", "Image", "Image", "Image", "colorbar"],
                     ["channelGroup3", "Image", "Image", "Image", "."],
                    ]
    
    # This order is left to right, which we want to be top to bottom in the image, so reverse this
    newSensorOrder = ["T7", "F7", "P7", "FC5", "CP5", "F3", "C3", "P3", "Fp1", "AF3", "FC1", "CP1", "PO3", "O1", "Fz", "Cz", "Pz", "Oz", "Fp2", "AF4", "FC2", "CP2", "PO4", "O2", "F4", "C4", "P4", "FC6", "CP6", "F8", "T8", "P8"]
    # selections_new = {"All": np.concatenate((selections["Left"], selections["Midline"], selections["Right"]))}
    newSensorInds = []
    for i in range(len(newSensorOrder)):
        newSensorInds.append(evokedDiffEEG.ch_names.index(newSensorOrder[i]))
    selections_new = {"All": np.flip(np.array(newSensorInds))}
    figEegImage, axesEegImage = plt.subplot_mosaic(mosaic=mosaicGridEeg, figsize=(11, 12), height_ratios=[3, 3, 2, 2, 2], width_ratios=[1, 1, 1, 1, .3])
    # axesMegImageDict = {sel: ax for sel, ax in zip(selections_meg, axesMegImage.ravel())}
    # axesEegImageDict = {"Left": axesEegImage["Left"],
    #                     "Midline": axesEegImage["Midline"],
    #                     "Right": axesEegImage["Right"],
    #                    }
    axesEegImageDict = {"All": axesEegImage["Image"],
                       }
    evokedDiffEEG.plot_image(
        axes=axesEegImageDict,
        # axes=axesEegImage["Image"],
        group_by=selections_new,
        colorbar=False,
        show=False,
        mask=significant_points,
        mask_alpha=1,
        show_names="all",
        # show_names=False,
        titles=None,
        **time_unit_dict,
    )
    # plt.tight_layout()
    
    maxAbs = np.max(np.abs(evokedDiffEEG.get_data()))
    
    # Create divider, add cax, and create colorbar
    dividerColorbars = make_axes_locatable(axesEegImage["colorbar"])
    axesEegImage["colorbar"].set_axis_off()
    cax = dividerColorbars.append_axes("right", size=.5, pad=0.05)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-maxAbs, vmax=maxAbs))
    plt.colorbar(sm, cax=cax)
    cax2 = dividerColorbars.append_axes("right", size=.5, pad=0.05)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=-maxAbs, vmax=maxAbs))
    plt.colorbar(sm, cax=cax2, label="Z-scored potential difference")
    # plt.colorbar(axesMegImageDict["Upper Left"].images[-1], ax=list(axesMegImageDict.values()), shrink=0.3, label="Z-scored potential difference")
    
    # cax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
    
    evokedDiffEEG.plot(time_unit=time_unit, 
                       axes=axesEegImage["butterfly"], 
                       titles=dict(eeg=""), 
                       units="Z-scored potential difference", 
                       zorder="std",
                       scalings=dict(eeg=1, mag=1, grad=1),
                      )
    axesEegImage["butterfly"].xaxis.set_label_text("")
    
    ymin_butterfly, ymax_butterfly = axesEegImage["butterfly"].get_ylim()
    
    if custLines:
        for line in axesEegImage["butterfly"].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
    
    for axisHere in axesEegImage:
        for text_obj in axesEegImage[axisHere].texts:
            if "ave" in text_obj.get_text():
                text_obj.set_visible(False)
                # or remove it from the list
                # ax.texts.remove(text_obj)
                # break # Assuming only one such text
    
    # timesHereMeans_meg = []
    # for i, clusterInd in enumerate(sigClusterInds_meg):
    #     timesHere = np.where(clusters_meg[clusterInd])[0] / 5000
    #     timesHereMeans_meg.append(np.mean(timesHere))
    
    # diffJointMEG = evokedDiffMEG.plot_joint(
    #     title="Speech vs. click", ts_args=time_unit_dict, topomap_args=time_unit_dict, times=np.sort(timesHereMeans_meg), 
    # )  # show difference wave
    # if custLines:
    #     for line in diffJointMEG.axes[3].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
    #         line.set_linewidth(jointLinewidth)
    #         line.set_alpha(jointAlpha)
    
    plt.show()
    
    
    eegClusterTimes = []
    
    # loop over clusters
    for i_clu, clu_idx in enumerate(sigClusterInds_eeg[np.argsort(timesHereMeans)][0:3]):
        # unpack cluster information, get unique indices
        # time_inds, space_inds = np.squeeze(clusters_meg[clu_idx])
        time_inds, space_inds = np.where(clusters_eeg[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
    
        # # This takes just the middle third of the total cluster time period
        # firstNewInd = int(np.floor(len(time_inds) / 3))
        # secondNewInd = int(np.floor(2 * len(time_inds) / 3))
        # time_inds = time_inds[firstNewInd:secondNewInd]
    
        # get topography for F stat
        # t_map = np.nanmean(t_obs_eeg[time_inds, ...], axis=0)
        t_map = np.nanmean(evokedDiffEEG.get_data().T[time_inds, ...], axis=0)
    
        # get signals at the sensors contributing to the cluster
        sig_times = evokedsEeg[0].times[time_inds]
        if time_unit == "ms":
            sig_times *= 1000
    
        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
    
        # initialize figure
        fig_topo_eeg, ax_topo_eeg = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")
    
        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], evokedsEeg[0].info, tmin=0)
        maxAbs = np.max(np.abs(t_evoked.get_data()))
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo_eeg,
            # cmap="Reds",
            # cmap="RdBu_r",
            cmap="coolwarm",
            vlim=(-maxAbs, maxAbs),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
            scalings=dict(eeg=1, mag=1, grad=1),
        )
        image_eeg = ax_topo_eeg.images[0]
    
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=axesEegImage[topoNames[i_clu]],
            # cmap="Reds",
            # cmap="RdBu_r",
            cmap="coolwarm",
            vlim=(-maxAbs, maxAbs),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
            scalings=dict(eeg=1, mag=1, grad=1),
        )
        image2_eeg = axesEegImage[topoNames[i_clu]].images[0]
    
        # remove the title that would otherwise say "0.000 s"
        ax_topo_eeg.set_title("")
        axesEegImage[topoNames[i_clu]].set_title("")
    
        # create additional axes (for ERF and colorbar)
        divider_eeg = make_axes_locatable(ax_topo_eeg)
        divider2_eeg = make_axes_locatable(axesEegImage[topoNames[i_clu]])
    
        # add axes for colorbar
        ax_colorbar_eeg = divider_eeg.append_axes("right", size="5%", pad=0.05)
        ax_colorbar2_eeg = divider2_eeg.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image_eeg, cax=ax_colorbar_eeg, label="t value")
        ax_topo_eeg.set_xlabel(
            "Averaged t map ({:0.1f} - {:0.1f} ms)".format(*sig_times[[0, -1]])
        )
        plt.colorbar(image2_eeg, cax=ax_colorbar2_eeg, label="t value")
        axesEegImage[topoNames[i_clu]].set_xlabel(
            "Averaged t map ({:0.1f} - {:0.1f} ms)".format(*sig_times[[0, -1]])
        )
    
        # add new axis for time courses and plot time courses
        ax_signals_eeg = divider_eeg.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean)"
        mne.viz.plot_compare_evokeds(
            evokedsEeg,
            title=title,
            picks=ch_inds,
            axes=ax_signals_eeg,
            colors=colors,
            # linestyles=linestyles,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
            time_unit=time_unit,
        )
    
        # plot temporal cluster extent
    
        eegClusterTimes.append(np.array([sig_times[0], sig_times[-1]]))
        
        ymin, ymax = ax_signals_eeg.get_ylim()
        ax_signals_eeg.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )
        axesEegImage["butterfly"].fill_betweenx(
            (ymin_butterfly, ymax_butterfly), sig_times[0], sig_times[-1], color="silver", alpha=0.75
        )
    
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections["Left"]], axes=axesEegImage["channelGroup1"], linewidth=0)
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections["Midline"]], axes=axesEegImage["channelGroup2"], linewidth=0)
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections["Right"]], axes=axesEegImage["channelGroup3"], linewidth=0)
    figEegImage.tight_layout()
    
    
    # for i, clusterInd in enumerate(sigClusterInds_meg):
    #     timesHere = np.where(clusters_meg[clusterInd])[0] / 5000
    #     print(f"MEG cluster {i+1}: Times {timesHere[0]} to {timesHere[-1]}; p = {p_values_meg[clusterInd]}")

# %%
if doClusters:

    # We subselect clusters that we consider significant at an arbitrarily
    # picked alpha level: "p_accept".
    # NOTE: remember the caveats with respect to "significant" clusters that
    # we mentioned in the introduction of this tutorial!
    
    sigClusterInds_meg = np.where(p_values_meg < p_accept)[0]
    
    # configure variables for visualization
    colors = {"M: All (real)": "crimson", "T: Clicks complete (real)": "steelblue"}
    linestyles = {"M: All (real)": "-", "T: Clicks complete (real)": "--"}
    
    # organize data for plotting
    # evokeds = {cond: epochs[cond].average() for cond in event_id}
    evokedsMeg = [speechMEGnewEvoked, clicksMEGnewEvoked]
    
    
    evokedDiffMEG = mne.combine_evoked(
        [speechMEGnewEvoked, clicksMEGnewEvoked], weights=[1, -1]
    )  # calculate difference wave
    
    layout = mne.channels.find_layout(speechMEGnewEvoked.info, ch_type="meg")
    
    # Create ROIs by checking channel labels
    # selections = mne.channels.make_1020_channel_selections(evokedDiffEEG.info, midline="12z")
    leftSensors = layout.pos[:, 0] < 0.45
    rightSensors = layout.pos[:, 0] >= 0.45
    lowerSensors = layout.pos[:, 1] < 0.55
    upperSensors = layout.pos[:, 1] >= 0.55
    selections_meg = {"Upper Right": np.where(rightSensors & upperSensors)[0],
                      "Lower Right": np.where(rightSensors & lowerSensors)[0],
                      "Lower Left": np.where(leftSensors & lowerSensors)[0],
                      "Upper Left": np.where(leftSensors & upperSensors)[0],
                     }
    significant_points_meg = clusters_meg[sigClusterInds_meg[0]]
    for i in sigClusterInds_meg[1:]:
        significant_points_meg = significant_points_meg | clusters_meg[i]
    
    significant_points_meg = significant_points_meg.T
    
    # Visualize the results
    # figMegImage, axesMegImage = plt.subplots(nrows=4, figsize=(8, 8))
    topoNames = ["topo1", "topo2", "topo3"]
    # mosaicGridMeg = [[".", topoNames[0], topoNames[1], topoNames[2], "."],
    #                  [".", "butterfly", "butterfly", "butterfly", "."],
    #                  ["channelGroup1", "Upper Left", "Upper Left", "Upper Left", "."],
    #                  ["channelGroup2", "Lower Left", "Lower Left", "Lower Left", "colorbar"],
    #                  ["channelGroup3", "Upper Right", "Upper Right", "Upper Right", "colorbar"],
    #                  ["channelGroup4", "Lower Right", "Lower Right", "Lower Right", "."],
    #                 ]
    
    mosaicGridMeg = [[".", topoNames[0], topoNames[1], topoNames[2], "."],
                     [".", "butterfly", "butterfly", "butterfly", "."],
                     ["channelGroup1", "Image", "Image", "Image", "."],
                     ["channelGroup2", "Image", "Image", "Image", "colorbar"],
                     ["channelGroup3", "Image", "Image", "Image", "colorbar"],
                     ["channelGroup4", "Image", "Image", "Image", "."],
                    ]
    figMegImage, axesMegImage = plt.subplot_mosaic(mosaic=mosaicGridMeg, figsize=(11, 12), height_ratios=[3, 3, 2, 2, 2, 2], width_ratios=[1, 1, 1, 1, .3])
    # axesMegImageDict = {sel: ax for sel, ax in zip(selections_meg, axesMegImage.ravel())}
    # axesMegImageDict = {"Upper Left": axesMegImage["Upper Left"],
    #                     "Lower Left": axesMegImage["Lower Left"],
    #                     "Upper Right": axesMegImage["Upper Right"],
    #                     "Lower Right": axesMegImage["Lower Right"]
    #                    }
    axesMegImageDict = {"All": axesMegImage["Image"]}
    selections_meg_new = {"All": np.flip(np.concatenate((selections_meg["Upper Right"], selections_meg["Lower Right"], selections_meg["Lower Left"], selections_meg["Upper Left"])))
                       }
    
    evokedDiffMEG.plot_image(
        axes=axesMegImageDict,
        group_by=selections_meg_new,
        colorbar=False,
        show=False,
        mask=significant_points_meg,
        mask_alpha=1,
        # show_names="all",
        show_names=False,
        titles=None,
        **time_unit_dict,
    )
    
    maxAbs = np.max(np.abs(evokedDiffMEG.get_data()))
    
    # Create divider, add cax, and create colorbar
    dividerColorbars = make_axes_locatable(axesMegImage["colorbar"])
    axesMegImage["colorbar"].set_axis_off()
    cax = dividerColorbars.append_axes("right", size=.5, pad=0.05)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-maxAbs, vmax=maxAbs))
    plt.colorbar(sm, cax=cax)
    cax2 = dividerColorbars.append_axes("right", size=.5, pad=0.05)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=-maxAbs, vmax=maxAbs))
    plt.colorbar(sm, cax=cax2, label="Z-scored field strength difference")
    
    # plt.tight_layout()
    # plt.colorbar(axesMegImageDict["Upper Left"].images[-1], ax=list(axesMegImageDict.values()), shrink=0.3, label="Z-scored field strength")
    
    evokedDiffMEG.plot(time_unit=time_unit, 
                       axes=axesMegImage["butterfly"], 
                       titles=dict(mag=""), 
                       units="Z-scored field strength difference", 
                       zorder="std",
                       scalings=dict(eeg=1, mag=1, grad=1),
                      )
    axesMegImage["butterfly"].xaxis.set_label_text("")
    
    ymin_butterfly, ymax_butterfly = axesMegImage["butterfly"].get_ylim()
    
    if custLines:
        for line in axesMegImage["butterfly"].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
    
    for axisHere in axesMegImage:
        for text_obj in axesMegImage[axisHere].texts:
            if "ave" in text_obj.get_text():
                text_obj.set_visible(False)
                # or remove it from the list
                # ax.texts.remove(text_obj)
                # break # Assuming only one such text
    
    timesHereMeans_meg = []
    for i, clusterInd in enumerate(sigClusterInds_meg):
        timesHere = np.where(clusters_meg[clusterInd])[0] / 5000
        timesHereMeans_meg.append(np.mean(timesHere))
    
    diffJointMEG = evokedDiffMEG.plot_joint(
        title="Speech vs. click", ts_args=time_unit_dict, topomap_args=time_unit_dict, times=np.sort(timesHereMeans_meg), 
    )  # show difference wave
    if custLines:
        for line in diffJointMEG.axes[3].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
            line.set_linewidth(jointLinewidth)
            line.set_alpha(jointAlpha)
    
    plt.show()
    
    
    megClusterTimes = []
    
    # loop over clusters
    for i_clu, clu_idx in enumerate(sigClusterInds_meg[np.argsort(timesHereMeans_meg)][0:3]):
        # unpack cluster information, get unique indices
        # time_inds, space_inds = np.squeeze(clusters_meg[clu_idx])
        time_inds, space_inds = np.where(clusters_meg[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
    
        # This takes just the middle third of the total cluster time period
        # firstNewInd = int(np.floor(len(time_inds) / 3))
        # secondNewInd = int(np.floor(2 * len(time_inds) / 3))
        # time_inds = time_inds[firstNewInd:secondNewInd]
    
        # get topography for F stat
        t_map = t_obs_meg[time_inds, ...].mean(axis=0)
    
        # get signals at the sensors contributing to the cluster
        sig_times = evokedsMeg[0].times[time_inds]
        if time_unit == "ms":
            sig_times *= 1000
    
        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
    
        # initialize figure
        fig_topo, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")
    
        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], evokedsMeg[0].info, tmin=0)
        maxAbs = np.max(np.abs(t_evoked.get_data()))
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            # cmap="Reds",
            # cmap="RdBu_r",
            cmap="coolwarm",
            vlim=(-maxAbs, maxAbs),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
            scalings=dict(eeg=1, mag=1, grad=1),
        )
        image = ax_topo.images[0]
    
        t_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=axesMegImage[topoNames[i_clu]],
            # cmap="Reds",
            # cmap="RdBu_r",
            cmap="coolwarm",
            vlim=(-maxAbs, maxAbs),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
            scalings=dict(eeg=1, mag=1, grad=1),
        )
        image2 = axesMegImage[topoNames[i_clu]].images[0]
    
        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")
        axesMegImage[topoNames[i_clu]].set_title("")
    
        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)
        divider2 = make_axes_locatable(axesMegImage[topoNames[i_clu]])
    
        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        ax_colorbar2 = divider2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar, label="t value")
        ax_topo.set_xlabel(
            "Averaged t map ({:0.1f} - {:0.1f} ms)".format(*sig_times[[0, -1]])
        )
        plt.colorbar(image2, cax=ax_colorbar2, label="t value")
        axesMegImage[topoNames[i_clu]].set_xlabel(
            "Averaged t map ({:0.1f} - {:0.1f} ms)".format(*sig_times[[0, -1]])
        )
    
        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean)"
        mne.viz.plot_compare_evokeds(
            evokedsMeg,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            colors=colors,
            # linestyles=linestyles,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
            time_unit=time_unit,
        )
    
        megClusterTimes.append(np.array([sig_times[0], sig_times[-1]]))
    
        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )
        axesMegImage["butterfly"].fill_betweenx(
            (ymin_butterfly, ymax_butterfly), sig_times[0], sig_times[-1], color="silver", alpha=0.75
        )
    
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections_meg["Upper Right"]], axes=axesMegImage["channelGroup1"], linewidth=0)
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections_meg["Lower Right"]], axes=axesMegImage["channelGroup2"], linewidth=0)
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections_meg["Lower Left"]], axes=axesMegImage["channelGroup3"], linewidth=0)
    mne.viz.plot_sensors(t_evoked.info, ch_groups=[selections_meg["Upper Left"]], axes=axesMegImage["channelGroup4"], linewidth=0)
    figMegImage.tight_layout()
    
    
    for i, clusterInd in enumerate(sigClusterInds_meg):
        timesHere = np.where(clusters_meg[clusterInd])[0] / 5000
        chansHere = np.where(clusters_meg[clusterInd])[1]
        print(f"MEG cluster {i+1}: Times {timesHere[0]} to {timesHere[-1]}; p = {p_values_meg[clusterInd]}")
        print(f"MEG cluster {i+1}: Number of channels is {len(np.unique(chansHere))}")

# %%
if doPCplot2:

    # Start with 4 datasets
    dataForPCs = []
    dataForPCs.append(condsToPlot[1][0][0].copy())
    dataForPCs.append(condsToPlot[2][0][0].copy())
    dataForPCs.append(condsToPlot[3][0][0].copy())
    dataForPCs.append(condsToPlot[4][0][0].copy())
    # dataForPCs.append(evokedAvgTargetAll[0][0].copy())
    # dataForPCs.append(evokedAvgDistAll[0][0].copy())
    # dataForPCs.append(evokedAvgMixAll[0][0].copy())
    # dataForPCs.append(evokedAvgClicksComplete[0][0].copy())
    # dataForPCs.append(evokedAvgTargetMale[0][0].copy())
    # dataForPCs.append(evokedAvgDistMale[0][0].copy())
    # dataForPCs.append(evokedAvgTargetFemale[0][0].copy())
    # dataForPCs.append(evokedAvgDistFemale[0][0].copy())
    
    # pcsEEGtarget = []
    # pcsEEGdist = []
    # pcsEEGspeech = []
    # pcsEEGclicks = []
    pcsEEG = [[] for _ in range(len(dataForPCs))]
    
    eegClusterTimes_zeroLocked, megClusterTimes_zeroLocked = eb.load.unpickle("/Users/karl/Dropbox/UMD/ABR_MLR_TRF/timeZeroLockedClusterTimes.pickle")
    
    n_lines = 7
    pcAlpha = 0.75
    pcMinAlpha = 0.3
    pcAlphas = np.linspace(pcAlpha, pcMinAlpha, n_lines)
    
    times = dataForPCs[0].times * 1000
    
    pcFig, ((pcAxEEG, pcAxMEG), (pcAxEEG2, pcAxMEG2)) = plt.subplots(2, 2, figsize=(12, 7))
    for i in range(len(pcsEEG)):
        pcsEEG[i].append(calcPC(sp.stats.zscore(dataForPCs[i].data[np.newaxis, 157:, :], axis=None, nan_policy="omit"), True, None, whichPC=0).squeeze())
    for j in range(1, n_lines):
        for i in range(len(pcsEEG)):
            pcsEEG[i].append(calcPC(sp.stats.zscore(dataForPCs[i].data[np.newaxis, 157:, :], axis=None, nan_policy="omit"), True, pcsEEG[i][0], whichPC=j).squeeze())
    for i in range(len(pcsEEG)):
        pcsEEG[i] = np.array(pcsEEG[i])
    
    firstPcCmap = matplotlib.colormaps["Purples_r"]
    secondPcCmap = matplotlib.colormaps["Greens_r"]
    
    # targetPcColors = np.flip(targetPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    # distPcColors = np.flip(distPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    firstPcColors = firstPcCmap(np.linspace(0, 1, n_lines))
    secondPcColors = secondPcCmap(np.linspace(0, 1, n_lines))
    # pcFig, pcAx = plt.subplots()
    
    # pcsEEGtarget = np.flip(pcsEEGtarget, axis=0)
    # pcsEEGdist = np.flip(pcsEEGdist, axis=0)
    
    
    for i in range(n_lines):
        pcAxEEG.plot(times, pcsEEG[0][i], color=firstPcColors[i], linewidth=3, alpha=pcAlphas[i])
        pcAxEEG.plot(times, pcsEEG[1][i], color=secondPcColors[i], linewidth=3, alpha=pcAlphas[i])
    
    pcAxEEG.set_yticks([])
    pcAxEEG.set_title("EEG: Target Easy vs. Distractor Hard")
    
    sm = plt.cm.ScalarMappable(cmap="Greens_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxEEG, label="Dist Hard PC number")
    sm = plt.cm.ScalarMappable(cmap="Purples_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxEEG, label="Targ Easy PC number")
    
    thirdPcCmap = matplotlib.colormaps["Blues_r"]
    fourthPcCmap = matplotlib.colormaps["Oranges_r"]
    
    # speechPcColors = np.flip(speechPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    # clicksPcColors = np.flip(clicksPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    thirdPcColors = thirdPcCmap(np.linspace(0, 1, n_lines))
    fourthPcColors = fourthPcCmap(np.linspace(0, 1, n_lines))
    # pcFig2, pcAx2 = plt.subplots()
    
    # pcsEEGspeech = np.flip(pcsEEGspeech, axis=0)
    # pcsEEGclicks = np.flip(pcsEEGclicks, axis=0)
    
    
    for i in range(n_lines):
        pcAxEEG2.plot(times, pcsEEG[2][i], color=thirdPcColors[i], linewidth=3, alpha=pcAlphas[i])
        pcAxEEG2.plot(times, pcsEEG[3][i], color=fourthPcColors[i], linewidth=3, alpha=pcAlphas[i])
    
    pcAxEEG2.set_yticks([])
    pcAxEEG2.set_xlabel("Time (ms)")
    pcAxEEG2.set_title("EEG: Target Hard vs. Distractor Easy")
    
    # maxAbs = np.max((np.max(np.abs(pcsEEGspeech)), np.max(np.abs(pcsEEGclicks))))
    # for i, eegClusterTime in enumerate(eegClusterTimes):
    #     pcAxEEG2.fill_betweenx((-maxAbs, maxAbs), eegClusterTime[0], eegClusterTime[1], color="silver", alpha=0.5)
    # for i, eegClusterTime in enumerate(eegClusterTimes_zeroLocked):
    #     pcAxEEG2.fill_betweenx((-maxAbs*0.7, maxAbs*0.7), eegClusterTime[0], eegClusterTime[1], color="pink", alpha=0.5)
    
    sm = plt.cm.ScalarMappable(cmap="Oranges_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxEEG2, label="Dist Easy PC number")
    sm = plt.cm.ScalarMappable(cmap="Blues_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxEEG2, label="Targ Hard PC number")
    
    
    pcsMEG = [[] for _ in range(len(dataForPCs))]    

    for i in range(len(pcsMEG)):
        pcsMEG[i].append(calcPC(sp.stats.zscore(dataForPCs[i].data[np.newaxis, :157, :], axis=None, nan_policy="omit"), True, pcsEEG[i][0], whichPC=0).squeeze())
    for j in range(1, n_lines):
        for i in range(len(pcsMEG)):
            pcsMEG[i].append(calcPC(sp.stats.zscore(dataForPCs[i].data[np.newaxis, :157, :], axis=None, nan_policy="omit"), True, pcsMEG[i][0], whichPC=j).squeeze())
    for i in range(len(pcsMEG)):
        pcsMEG[i] = np.array(pcsMEG[i])
    
    # targetPcCmap = matplotlib.colormaps["Purples_r"]
    # distPcCmap = matplotlib.colormaps["Greens_r"]
    # targetPcColors = np.flip(targetPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    # distPcColors = np.flip(distPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    # pcFigMeg, pcAxMeg = plt.subplots()
    
    # pcsMEGtarget = np.flip(pcsMEGtarget, axis=0)
    # pcsMEGdist = np.flip(pcsMEGdist, axis=0)
    
    
    for i in range(n_lines):
        pcAxMEG.plot(times, pcsMEG[0][i], color=firstPcColors[i], linewidth=3, alpha=pcAlphas[i])
        pcAxMEG.plot(times, pcsMEG[1][i], color=secondPcColors[i], linewidth=3, alpha=pcAlphas[i])
    
    pcAxMEG.set_yticks([])
    pcAxMEG.set_title("MEG: Target Easy vs. Distractor hard")
    
    sm = plt.cm.ScalarMappable(cmap="Greens_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxMEG, label="Dist Hard PC number")
    sm = plt.cm.ScalarMappable(cmap="Purples_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxMEG, label="Targ Easy PC number")
    # speechPcCmap = matplotlib.colormaps["Blues_r"]
    # clicksPcCmap = matplotlib.colormaps["Oranges_r"]
    # speechPcColors = np.flip(speechPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    # clicksPcColors = np.flip(clicksPcCmap(np.linspace(0, 1, n_lines)), axis=0)
    # pcFigMeg2, pcAxMeg2 = plt.subplots()
    
    # pcsMEGspeech = np.flip(pcsMEGspeech, axis=0)
    # pcsMEGclicks = np.flip(pcsMEGclicks, axis=0)
    
    
    for i in range(n_lines):
        pcAxMEG2.plot(times, pcsMEG[2][i], color=thirdPcColors[i], linewidth=3, alpha=pcAlphas[i])
        pcAxMEG2.plot(times, pcsMEG[3][i], color=fourthPcColors[i], linewidth=3, alpha=pcAlphas[i])
    
    pcAxMEG2.set_yticks([])
    pcAxMEG2.set_xlabel("Time (ms)")
    pcAxMEG2.set_title("MEG: Target Hard vs. Distractor Easy")
    
    # maxAbs = np.max((np.max(np.abs(pcsMEGspeech)), np.max(np.abs(pcsMEGclicks))))
    # for i, megClusterTime in enumerate(megClusterTimes):
    #     pcAxMEG2.fill_betweenx((-maxAbs, maxAbs), megClusterTime[0], megClusterTime[1], color="silver", alpha=0.5)
    # for i, megClusterTime in enumerate(megClusterTimes_zeroLocked):
    #     pcAxMEG2.fill_betweenx((-maxAbs*0.7, maxAbs*0.7), megClusterTime[0], megClusterTime[1], color="pink", alpha=0.5)
        
    sm = plt.cm.ScalarMappable(cmap="Oranges_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxMEG2, label="Dist Easy PC number")
    sm = plt.cm.ScalarMappable(cmap="Blues_r", norm=plt.Normalize(vmin=n_lines, vmax=1))
    plt.colorbar(sm, ax=pcAxMEG2, label="Targ Hard PC number")
    
    pcFig.tight_layout()

# %%
# plt.show(block=True)

# %%

# %%

# %%
