# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
import joblib
import os
import sys
import pickle
import glob
import time

from prepTrfObjs import *

matplotlib.use("QtAgg")
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

# filePrefix = "anMeegRecFieldAllFolds"
# filePrefix = "anRecFieldAllFolds"


pickleSuffix = ".pickle"
# pickleSuffix = ".pickle.LAPL"
# pickleSuffix = ".pickle.ORIG"
# pickleSuffix = ".pickle.WEIRD"

# edgePad = 0.001

useDefaultMontage = True
# useDefaultMontage = False



shiftEvoked = None
subjects_dir = os.path.expandvars("$SUBJECTS_DIR")


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


# nameOfRegressor = "_ANmodel_fs5000"
# lenResponse = 425
# lenToRamp = .01
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


# nameOfRegressors = [
#     "_ANmodel_fs5000",
#     "_glottalPulses_fs5000",
# ]

optimizeLambda = False

# lambdas = np.logspace(3, 11, 15)
# lambdaInd = 8  # This indexes the lambdas that have been run, defined above

lambdas = np.array([0, 1e4, 1e7, 1e10])
lambdaInd = 0  # This indexes the lambdas that have been run, defined above

zScoring = False


timeToAddToStart = 0
# timeToAddToStart = 0.008

complexSubtypeNames = [" (real)", " (imag)", " (abs)", " (phase)"]

ci = 0.95  # confidence interval for shading



cutoffFreq = 300
filterTRFs = False
# sos = sp.signal.butter(3, cutoffFreq, output="sos")

resetAllBads = False
nanOutBads = False
normalizeByDataSTD = True

doSensorPlotting = False
doSourcePlotting = False
doSourceSaving = True
doSources = True
doDipoles = True
doMorphs = True

# subjectsToAverageAll = [[
#     "R3045",
#     "R3089",
#     "R3093",
#     "R3095",
#     "R3151",
#     "R2774",
#     "R3152",
#     "R2877",
#     "R3157",
#     "R2783",
#     "R3170",
#     "R3172",
#     "R3193",
#     "R3184",
#     "R3214",
#     "R3265",
#     "R3275",
#     "R3247",
#     "R3292",
#     "R3293",
#     "R3294",
#     "R3303",
#     "R3304",
#     "R3306",
#     "R3313",
#     "R3317",
#     "R3320",
# ]]

subjectsToAverageAll = [
    ["R3045"],
    ["R3089"],
    ["R3093"],
    ["R3095"],
    ["R3151"],
    ["R2774"],
    ["R3152"],
    ["R2877"],
    ["R3157"],
    ["R2783"],
    ["R3170"],
    ["R3172"],
    ["R3193"],
    ["R3184"],
    ["R3214"],
    ["R3265"],
    ["R3275"],
    ["R3247"],
    ["R3292"],
    ["R3293"],
    ["R3294"],
    ["R3303"],
    ["R3304"],
    ["R3306"],
    ["R3313"],
    ["R3317"],
    ["R3320"],
]

additionalBadChansAll = None
    
clickList = ["R3265", "R3275", "R3247", "R3292", "R3293", "R3294", "R3303", "R3304", "R3306", "R3313", "R3317", "R3320"]

# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877", "R3157"]
# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877"]
# subjectsToAverage=["R3095", "R3151", "R2774", "R3152", "R2877", "R3157", "R2783"]
# subjectsToAverage=["R2877", "R3151", "R3152"]

# subjectsToAverage = ["R3157"]
# condIndForSources = 2

# subset = [17]
subset = [i for i in range(len(subjectsToAverageAll))]
# subset = np.arange(19, 27)

subjectsToAverageAll = [subjectsToAverageAll[i] for i in subset]
if additionalBadChansAll is not None:
    additionalBadChansAll = [additionalBadChansAll[i] for i in subset]

# condIndForSources = 0

multFactor = 1

depth = 0.8

# eegOnly = True

sourceModes = ["volOnly", "mixed"]
oriModes = ["vector", None]

avgBrain = "fsaverage"
# avgBrain = "mni"

if doSources:
    src_fs = []
    bem_fs = eb.load.unpickle(f"{subjects_dir}/{avgBrain}/bem/conductorModel.pickle")
    for sourceMode in sourceModes:
        
        if sourceMode == "volOnly":
            
            volume_label = None
            
            src_fs_here = mne.setup_volume_source_space(
                avgBrain,
                mri=subjects_dir + f"/{avgBrain}/mri/aparc.a2009s+aseg.mgz",
                pos=8.0,
                bem=bem_fs,
                volume_label=volume_label,
                subjects_dir=subjects_dir,
                add_interpolator=True,  # just for speed, usually this should be True
                verbose=True,
            )
            src_fs.append(src_fs_here)
            
        elif sourceMode == "mixed":
            
            volume_label = [
                "Left-Thalamus-Proper",
                "Right-Thalamus-Proper",
                "Brain-Stem",
            ]
            
            src_surf_fs = mne.setup_source_space(
                subject=avgBrain, spacing="oct5", add_dist=False, subjects_dir=subjects_dir
            )    
            src_vol_fs = mne.setup_volume_source_space(
                avgBrain,
                mri=subjects_dir + f"/{avgBrain}/mri/aseg.mgz",
                pos=8.0,
                bem=bem_fs,
                volume_label=volume_label,
                subjects_dir=subjects_dir,
                add_interpolator=True,  # just for speed, usually this should be True
                verbose=True,
            )
            src_fs_here = src_surf_fs + src_vol_fs
            src_fs.append(src_fs_here)

# %%
timesFiveAll = []

for subjInd, subjectsToAverage in enumerate(subjectsToAverageAll):

    if additionalBadChansAll is not None:
        additionalBadChans = additionalBadChansAll[subjInd]
    else:
        additionalBadChans = additionalBadChansAll

    filePrefix = "anMeegRecFieldAllFolds"
    
    if filePrefix == "anRecFieldAllFolds":
        subDirs = "/eegAndMeg/eeg/"
        nChannels = 32
        eegOnly = True
    else:
        subDirs = "/eegAndMeg/meg/"
        nChannels = 32 + 157
        eegOnly = False

    nameOfRegressor = "_ANmodel_fs5000"
    lenResponse = 500
    lenToRamp = 0
    expToRamp = 4
    fs = 5000
    nameOfRegressors = [
        "_ANmodel_fs5000",
        "_glottalPulses_fs5000",
    ]
    
    rampFunction = np.concatenate(
        (
            np.linspace(0, 1, int(lenToRamp * fs)) ** expToRamp,
            np.ones(lenResponse - int(lenToRamp * fs)),
        )
    )
    rampFunction = rampFunction[None, :]

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
                              additionalBadChans,
                              multFactor=multFactor,
                             )
    
    # typeOfRegressor = "mix"
    # filenameSuffix = "_easy"
    # comment = "M: Easy"
    
    # evokedAvgMixEasy = loadAndPrepareEvokeds(typeOfRegressor, 
    #                           filenameSuffix, 
    #                           lenResponse, 
    #                           nChannels, 
    #                           subjectsToAverage, 
    #                           parentDir, 
    #                           subDirs, 
    #                           filePrefix, 
    #                           lambdas, 
    #                           lambdaInd, 
    #                           optimizeLambda, 
    #                           nameOfRegressor, 
    #                           nameOfRegressors,
    #                           nanOutBads,
    #                           zScoring,
    #                           comment,
    #                           complexSubtypeNames,
    #                           pickleSuffix,
    #                           shiftEvoked,
    #                           normalizeByDataSTD,
    #                           useDefaultMontage,
    #                           filterTRFs,
    #                           cutoffFreq,
    #                           timeToAddToStart,
    #                           rampFunction,
    #                           resetAllBads,
    #                           additionalBadChans,
    #                          )
    
    # typeOfRegressor = "mix"
    # filenameSuffix = "_hard"
    # comment = "M: Hard"
    
    # evokedAvgMixHard = loadAndPrepareEvokeds(typeOfRegressor, 
    #                           filenameSuffix, 
    #                           lenResponse, 
    #                           nChannels, 
    #                           subjectsToAverage, 
    #                           parentDir, 
    #                           subDirs, 
    #                           filePrefix, 
    #                           lambdas, 
    #                           lambdaInd, 
    #                           optimizeLambda, 
    #                           nameOfRegressor, 
    #                           nameOfRegressors,
    #                           nanOutBads,
    #                           zScoring,
    #                           comment,
    #                           complexSubtypeNames,
    #                           pickleSuffix,
    #                           shiftEvoked,
    #                           normalizeByDataSTD,
    #                           useDefaultMontage,
    #                           filterTRFs,
    #                           cutoffFreq,
    #                           timeToAddToStart,
    #                           rampFunction,
    #                           resetAllBads,
    #                           additionalBadChans,
    #                          )
    
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
                              additionalBadChans,
                              multFactor=multFactor,
                             )
    
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
                              additionalBadChans,
                              multFactor=multFactor,
                             )

    typeOfRegressor = "mix"
    filenameSuffix = "_easy"    
    comment = "M: Easy"

    evokedAvgMixEasy = loadAndPrepareEvokeds(typeOfRegressor, 
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
                                 )

    typeOfRegressor = "mix"
    filenameSuffix = "_hard"    
    comment = "M: Hard"

    evokedAvgMixHard = loadAndPrepareEvokeds(typeOfRegressor, 
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
                                 )
    
    if subjectsToAverage[0] in clickList:
        # Redefine some of these variables here because they were defined at the top for speech maintask responses
        filePrefix = "abrAndTones"
        
        if filePrefix == "anRecFieldAllFolds":
            subDirs = "/eegAndMeg/eeg/"
            nChannels = 32
            eegOnly = True
        else:
            subDirs = "/eegAndMeg/meg/"
            nChannels = 32 + 157
            eegOnly = False
        
        nameOfRegressor = "_ANmodel_clicks"
        lenResponse = 500
        lenToRamp = 0
        expToRamp = 4
        fs = 5000
        nameOfRegressors = [
            "_ANmodel_clicks",
        ]
    
        rampFunction = np.concatenate(
            (
                np.linspace(0, 1, int(lenToRamp * fs)) ** expToRamp,
                np.ones(lenResponse - int(lenToRamp * fs)),
            )
        )
        rampFunction = rampFunction[None, :]
    
        typeOfRegressor = "clicksComplete"
        
        filenameSuffix = "_minimumPhase"
        
        comment = "T: Clicks complete"
        
        evokedAvgClicksComplete = loadAndPrepareEvokeds(typeOfRegressor, 
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
                                      additionalBadChans,
                                      multFactor=multFactor,
                                     )
    linewidth = 3
    alpha = 0.75
    avgOrAll = 0  # 0 for pre-averaged TRFs, 1 for lists of evoked objects for each subject
    
    complexSubtype = 0
    
    # condsToPlot = [evokedAvgQuiet, evokedAvgMixEasy, evokedAvgMixHard]
    
    if subjectsToAverage[0] in clickList:
        condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixEasy, evokedAvgMixHard, evokedAvgClicksComplete]
        colors = ["blue", "green", "red", "purple", "brown", "black"]
    else:
        condsToPlot = [evokedAvgQuiet, evokedAvgTargetAll, evokedAvgDistAll, evokedAvgMixEasy, evokedAvgMixHard]
        colors = ["blue", "green", "red", "purple", "brown"]
        
    # condsToPlot = [evokedAvgQuiet]
    
    keys = [cond[0][complexSubtype].comment for cond in condsToPlot]
    evokedValues = [cond[avgOrAll][complexSubtype] for cond in condsToPlot]
    evokeds = {k: v for (k, v) in zip(keys, evokedValues)}
    
    styleValues = [
        {"linewidth": linewidth, "alpha": alpha, "color": colors[i]}
        for i in range(len(condsToPlot))
    ]
    styles = {k: v for (k, v) in zip(keys, styleValues)}
    
    # mne.viz.plot_compare_evokeds(
    #     evokeds=evokeds, legend=True, axes="topo", styles=styles, ci=ci, picks="eeg"
    # )
    
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
        peakRanges = [[.0074, .0106], [.01, .0158], [.0245, .031], [.045, .06]]
        numPeaksPerRange = [1, 1, 1, 1]
        useEEGonly = [[True], [True], [True], [False]]
        eegSigns = [None, None, None, None]
        useMEGonly = [[False], [False], [False], [False]]
        order = 5
    else:
        peakRanges = [[.015, .07], [.07, .23], [.25, .7]]
        numPeaksPerRange = [1, 2, 2]
        useEEGonly = [[False], [False, False], [False, False]]
        useMEGonly = [[True], [True, True], [True, True]]
        # peakRanges = [[.015, .07], [.08, .12], [.13, .23]]
        # numPeaksPerRange = [1, 1, 1]
        # useEEGonly = [[True], [True], [True]]
        order = 10
    
    plotTimes = False
    timesFive = createPeakTimes(avgOrAll, condsToPlot, peakRanges, numPeaksPerRange, useEEGonly, useMEGonly, order, plotTimes, eegOnly, eegSigns)

    # This block is for lambda 0
    if subjectsToAverage[0] == "R3093":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "M: Hard (real)":
                timesFive[i] = np.array([0.0072, 0.0116, 0.0286, 0.0558])  # Adding a wave V for one condition
    elif subjectsToAverage[0] == "R3172":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment ==  "M: Hard (real)":
                timesFive[i] = np.array([0.0086, 0.0115, 0.0254, 0.0553])  # Correcting a wave 11
    elif subjectsToAverage[0] == "R3193": 
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "T: Quiet (real)":
                timesFive[i] = np.array([0.0104, 0.0127, 0.0306, 0.0512])  # Correcting a wave 11
            elif condsToPlot[i][0][0].comment ==  "D: All (real)":
                timesFive[i] = np.array([0.0102, 0.0128, 0.0292, 0.0571])  # Correcting a wave 11
    elif subjectsToAverage[0] == "R3275":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "T: Quiet (real)":
                timesFive[i] = np.array([0.009, 0.0138, 0.031, 0.0531])  # Correcting a Pa
            elif condsToPlot[i][0][0].comment == "T: Clicks complete (real)":
                timesFive[i] = np.array([0.0084, 0.014, 0.0306, 0.0469])  # Correcting a Pa
    elif subjectsToAverage[0] == "R3293": 
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "T: Quiet (real)":
                timesFive[i] = np.array([0.00756, 0.013, 0.0272, 0.0504])  # Adding a wave V
            elif condsToPlot[i][0][0].comment == "T: All (real)":
                timesFive[i] = np.array([0.00767, 0.0132, 0.0294, 0.0543])  # Adding a wave V
            elif condsToPlot[i][0][0].comment == "D: All (real)":
                timesFive[i] = np.array([0.00771, 0.013, 0.0276, 0.0495])  # Adding a wave V
            elif condsToPlot[i][0][0].comment == "M: Easy (real)":
                timesFive[i] = np.array([0.00756, 0.013, 0.0264, 0.0545])  # Adding a wave V
    elif subjectsToAverage[0] == "R3294":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "T: All (real)":
                timesFive[i] = np.array([0.0102, 0.0133, 0.0296, 0.0536])  # Adding a wave 11
    elif subjectsToAverage[0] == "R3304":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "M: Easy (real)":
                timesFive[i] = np.array([0.009, 0.0115, 0.0268, 0.0529])  # Correcting a wave 11
    elif subjectsToAverage[0] == "R3313":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "T: Quiet (real)":
                timesFive[i] = np.array([0.00784, 0.0136, 0.0272, 0.0554])  # Adding a wave V
            elif condsToPlot[i][0][0].comment == "T: Clicks complete (real)":
                timesFive[i] = np.array([0.008, 0.0161, 0.0286, 0.0509])  # Adding a wave 11
    elif subjectsToAverage[0] == "R3317":
        for i in range(len(condsToPlot)):
            if condsToPlot[i][0][0].comment == "T: Clicks complete (real)":
                timesFive[i] = np.array([0.0086, 0.01218, 0.0296, 0.051])  # Correcting a wave 11

    # This block is a quick hack for lambda 3 (1e10) with only wave V
    # if subjectsToAverage[0] == "R3093":
    #     for i in range(len(condsToPlot)):
    #         timesFive[i][0] = 0.0082  # Adding a wave V
    
    
    print("\n")
    print(f"Subject {subjectsToAverage}")
    print(timesFive)
    eb.save.pickle(timesFive, f"/Volumes/Seagate/map/{subjectsToAverage[0]}/timesFive_lambda{lambdaInd}.pickle")
    custLines = True
    jointLinewidth = 1.5
    jointAlpha = .75

    subject = subjectsToAverage[0]
    megLocation = parentDir + subject + subDirs
    
    if doSensorPlotting:
    
        for i, evoked in enumerate(evokedValues):
                    
            figJointEEG = evoked.plot_joint(picks="eeg", times=timesFive[i], title=evoked.comment, ts_args={"gfp": True})
        
            if custLines:
                for line in figJointEEG.axes[len(timesFive[i])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
                    line.set_linewidth(jointLinewidth)
                    line.set_alpha(jointAlpha)
                    
            if not eegOnly:
                    
                figJointMEG = evoked.plot_joint(picks="meg", times=timesFive[i], title=evoked.comment, ts_args={"gfp": True})
        
                if custLines:
                    for line in figJointMEG.axes[len(timesFive[i])].lines:  # npeaks as the index here because the first axes are the topo plots and lines, and then the main butterfly axis right after that
                        line.set_linewidth(jointLinewidth)
                        line.set_alpha(jointAlpha)
                        
    if doSources and len(subjectsToAverage) == 1:

        for evoked in evokedValues:
            evoked.set_eeg_reference("average", projection=True)
        infoForSources = evokedValues[0].info
        
        
        bem = eb.load.unpickle(f"{subjects_dir}/{subject}/bem/conductorModel.pickle")
        trans = f"{parentDir}{subject}/eegAndMeg/eeg/eeg-trans.fif"
        loose = "auto"
        # loose = dict(volume=1.0)
        # depth = 0.8
        # depth = 10.0

        for sourceModeInd, sourceMode in enumerate(sourceModes):

            if sourceMode == "volOnly":
                mriVol = subjects_dir + "/" + subject + "/mri/aparc.a2009s+aseg.mgz"
                volume_label = None
                src = mne.setup_volume_source_space(
                    subject,
                    mri=mriVol,
                    pos=8.0,
                    bem=bem,
                    volume_label=volume_label,
                    subjects_dir=subjects_dir,
                    add_interpolator=True,  # just for speed, usually this should be True
                    verbose=True,
                )
            elif sourceMode == "mixed":
                mriVol = subjects_dir + "/" + subject + "/mri/aseg.mgz"
                volume_label = [
                    "Left-Thalamus-Proper",
                    "Right-Thalamus-Proper",
                    "Brain-Stem",
                ]    
                src_surf = mne.setup_source_space(
                    subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir
                )    
                src_vol = mne.setup_volume_source_space(
                    subject,
                    mri=mriVol,
                    pos=8.0,
                    bem=bem,
                    volume_label=volume_label,
                    subjects_dir=subjects_dir,
                    add_interpolator=True,  # just for speed, usually this should be True
                    verbose=True,
                )
                src = src_surf + src_vol


            fwd = mne.make_forward_solution(
                infoForSources,
                trans,
                src,
                bem,
                mindist=5.0,  # ignore sources<=5mm from innerskull
                meg=True,
                eeg=True,
                n_jobs=None,
            )
            
    
            if doMorphs:
                morph = mne.compute_source_morph(
                    src=fwd["src"],
                    subject_from=subject,
                    subject_to=avgBrain,
                    src_to=src_fs[sourceModeInd],
                    subjects_dir=subjects_dir,
                    niter_sdr=[5, 5, 2],
                    niter_affine=[5, 5, 2],
                    zooms=7,  # just for speed
                    verbose=True,
                )
        
        
    
            print("\n")
            print(subject)
            print(timesFive)
            for condIndForSources in range(len(evokedValues)):
            
                if doSources and len(subjectsToAverage) == 1:

        
                    initial_time = timesFive[condIndForSources][0]
                    evokedHere = evokedValues[condIndForSources]
                    if evokedHere.comment == "T: Clicks complete (real)":
                        print("\n")
                        print(f"Got into clicks cov block, comment is {evokedHere.comment}")
                        print("\n")
                        noiseCov = eb.load.unpickle(f"{megLocation}highpassNoiseCovariance_abrAndTones.pickle")
                        dataCov = eb.load.unpickle(f"{megLocation}dataCovariance_abrAndTones_fs5000.pickle")
                    elif evokedHere.comment == "T: Quiet (real)" or evokedHere.comment == "T: All (real)" or evokedHere.comment == "D: All (real)" or evokedHere.comment == "M: Easy (real)" or evokedHere.comment == "M: Hard (real)":
                        print("\n")
                        print(f"Got into speech cov block, comment is {evokedHere.comment}")
                        print("\n")
                        noiseCov = eb.load.unpickle(f"{megLocation}highpassNoiseCovariance.pickle")
                        dataCov = eb.load.unpickle(f"{megLocation}dataCovariance_fs5000.pickle")
                    
                    inverse_operator = mne.minimum_norm.make_inverse_operator(
                        infoForSources, fwd, noiseCov, depth=depth, loose=loose, verbose=True
                    )
                    # snr = 3.0  # use smaller SNR for raw data
                    snr = 9
                    inv_method = "dSPM"  # sLORETA, MNE, dSPM
                
                    lambda2 = 1.0 / snr**2
    
                    for oriMode in oriModes:
                    
                        stc = mne.minimum_norm.apply_inverse(
                            evokedHere, inverse_operator, lambda2, inv_method, pick_ori=oriMode
                        )
            
                        if doMorphs:
                            stc_fs = morph.apply(stc)
            
                        dipAll = []
                        if doDipoles and sourceMode != "mixed":
                        
                            max_n_dipoles = 5
                            for n_dipoles in np.arange(1, max_n_dipoles+1):
                                dip = []
                                # figDip, axDip = plt.subplots(n_dipoles, 4, figsize=(25, 15), subplot_kw=dict(projection='3d'))
                                # initial_time = timesFive[condIndForSources][0]
                                tempTimes = None
                                # if subjectsToAverage[0] == "R3093" and len(timesFive[condIndForSources]) < np.array(numPeaksPerRange).sum():
                                #     tempTimes = [0.0072, 0.0116, 0.0286, 0.0558]  # Adding a wave V for one condition
                                # elif subjectsToAverage[0] == "R3172" and evokedHere.comment == "M: Hard (real)":
                                #     tempTimes = [0.0086, 0.0115, 0.0254, 0.0553]  # Correcting a wave 11
                                # elif subjectsToAverage[0] == "R3193" and evokedHere.comment == "T: Quiet (real)":
                                #     tempTimes = [0.0104, 0.0127, 0.0306, 0.0512]  # Correcting a wave 11
                                # elif subjectsToAverage[0] == "R3193" and evokedHere.comment == "D: All (real)":
                                #     tempTimes = [0.0102, 0.0128, 0.0292, 0.0571]  # Correcting a wave 11
                                # elif subjectsToAverage[0] == "R3275" and evokedHere.comment == "T: Quiet (real)":
                                #     tempTimes = [0.009, 0.0138, 0.031, 0.0531]  # Correcting a Pa
                                # elif subjectsToAverage[0] == "R3275" and evokedHere.comment == "T: Clicks complete (real)":
                                #     tempTimes = [0.0084, 0.014, 0.0306, 0.0469]  # Correcting a Pa
                                # elif subjectsToAverage[0] == "R3293" and evokedHere.comment == "T: Quiet (real)":
                                #     tempTimes = [0.00756, 0.013, 0.0272, 0.0504]  # Adding a wave V
                                # elif subjectsToAverage[0] == "R3293" and evokedHere.comment == "T: All (real)":
                                #     tempTimes = [0.00767, 0.0132, 0.0294, 0.0543]  # Adding a wave V
                                # elif subjectsToAverage[0] == "R3293" and evokedHere.comment == "D: All (real)":
                                #     tempTimes = [0.00771, 0.013, 0.0276, 0.0495]  # Adding a wave V
                                # elif subjectsToAverage[0] == "R3293" and evokedHere.comment == "M: Easy (real)":
                                #     tempTimes = [0.00756, 0.013, 0.0264, 0.0545]  # Adding a wave V
                                # elif subjectsToAverage[0] == "R3294" and evokedHere.comment == "T: All (real)":
                                #     tempTimes = [0.0102, 0.0133, 0.0296, 0.0536]  # Adding a wave 11
                                # elif subjectsToAverage[0] == "R3304" and evokedHere.comment == "M: Easy (real)":
                                #     tempTimes = [0.009, 0.0115, 0.0268, 0.0529]  # Correcting a wave 11
                                # elif subjectsToAverage[0] == "R3313" and evokedHere.comment == "T: Quiet (real)":
                                #     tempTimes = [0.00784, 0.0136, 0.0272, 0.0554]  # Adding a wave V
                                # elif subjectsToAverage[0] == "R3313" and evokedHere.comment == "T: Clicks complete (real)":
                                #     tempTimes = [0.008, 0.0161, 0.0286, 0.0509]  # Adding a wave 11
                                # elif subjectsToAverage[0] == "R3317" and evokedHere.comment == "T: Clicks complete (real)":
                                #     tempTimes = [0.0086, 0.01218, 0.0296, 0.051]  # Correcting a wave 11
                                for timeInd in range(1):
                                    # evokedForSources = evokedHere.copy().crop(tmin=timesFive[condIndForSources][timeInd]-.0002, tmax=timesFive[condIndForSources][timeInd]+.0002)
                                    if tempTimes is None:
                                        tminAndMax = timesFive[condIndForSources][timeInd]
                                    else:
                                        tminAndMax = tempTimes[timeInd]
                                        print("\n")
                                        print(f"Changing peak times for subject {subjectsToAverage[0]} from {timesFive[condIndForSources]} to {tempTimes}")
                                        print("\n")
                                    evokedForSources = evokedHere.copy().crop(tmin=tminAndMax, tmax=tminAndMax)
                            
                                    # dip1, res1 = mne.fit_dipole(evokedForSources, 
                                    #                           noiseCov,
                                    #                           bem, 
                                    #                           trans=trans)
                                    
                                    # dip2, res2 = mne.fit_dipole(res1, 
                                    #                           noiseCov,
                                    #                           bem, 
                                    #                           trans=trans)
                            
                                    
                                    dip.append(mne.beamformer.trap_music(evokedForSources, 
                                                                         fwd, 
                                                                         noiseCov, 
                                                                         n_dipoles=n_dipoles,
                                                                         return_residual=False,
                                                                        )
                                              )
                                    # for i in range(n_dipoles):
                                        # dip[timeInd][i].plot_locations(trans, subject, subjects_dir, mode="orthoview", ax=axDip[i, timeInd])
                                        # dip[timeInd][i].plot_locations(trans, subject, subjects_dir, mode="orthoview")
                                dipAll.append(dip)
                
                        # weight_norm = None
                        weight_norm = "unit-noise-gain-invariant"
                        # weight_norm = "nai"
                        depthBeam = depth
                        reg = 0.05
                        lcmv = mne.beamformer.make_lcmv(info=infoForSources, 
                                                        forward=fwd, 
                                                        data_cov=dataCov, 
                                                        reg=reg, 
                                                        noise_cov=noiseCov, 
                                                        label=None, 
                                                        pick_ori=oriMode, 
                                                        rank="info", 
                                                        weight_norm=weight_norm, 
                                                        reduce_rank=False, 
                                                        depth=depthBeam, 
                                                        inversion="matrix", 
                                                        verbose=None)
                        stcBeam = mne.beamformer.apply_lcmv(evokedHere, lcmv, verbose="info")

                        
                        if doMorphs:
                            stcBeam_fs = morph.apply(stcBeam)
                        
                        if doSourcePlotting:
                            brain = stc.plot_3d(
                                hemi="both",
                                src=src,
                                views="coronal",
                                initial_time=initial_time,
                                subjects_dir=subjects_dir,
                                brain_kwargs=dict(silhouette=True),
                                smoothing_steps=0,
                                show_traces=True,
                            )
                            if doMorphs:
                                brain2 = stc_fs.plot_3d(
                                    hemi="both",
                                    src=src_fs,
                                    views="coronal",
                                    initial_time=initial_time,
                                    subjects_dir=subjects_dir,
                                    brain_kwargs=dict(silhouette=True),
                                    smoothing_steps=0,
                                    show_traces=True,
                                )

                        if doSourceSaving:
                            if oriMode is None:
                                oriModeText = "None"
                            else:
                                oriModeText = oriMode
                            # sourceParamsSuffix = f"_depth{depth}"
                            # sourceParamsSuffix = f"_snr{snr}"
                            sourceParamsSuffix = f"_snr{snr}_invMethod_{inv_method}"
                            saveSuffix = f"_{sourceMode}_{oriModeText}_{avgBrain}.pickle"
                            eb.save.pickle((src, stc, stc_fs, dipAll, stcBeam, stcBeam_fs), 
                                           f"{megLocation}newSources_{evokedValues[condIndForSources].comment}_lambda{int(lambdaInd)}{sourceParamsSuffix}{saveSuffix}")

                    # This is just to make really sure we're opening new, correct covariances in the next loop iterations
                    del noiseCov
                    del dataCov

# %%

# %%
