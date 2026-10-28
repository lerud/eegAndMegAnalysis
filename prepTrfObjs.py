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


def readRecField(
    parentDir,
    subject,
    subDirs,
    filePrefix,
    typeOfRegressor,
    filenameSuffix,
    lambdas,
    lambdaInd,
    optimizeLambda,
    nameOfRegressor,
    nameOfRegressors,
    pickleSuffix,
    shiftEvoked,
    normalizeByDataSTD,
    useDefaultMontage,
    filterTRFs,
    cutoffFreq,
    timeToAddToStart,
    multFactor,
    cropEvoked,
):
    megLocation = parentDir + subject + subDirs
    regressorInd = nameOfRegressors.index(nameOfRegressor)

    if optimizeLambda:

        maxScores = {"MEG": np.zeros(len(lambdas)), "EEG": np.zeros(len(lambdas))}
        trfsAllLambdas = []

        for thisLambdaInd in range(len(lambdas)):

            temp = eb.load.unpickle(
                f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{thisLambdaInd}{pickleSuffix}"
            )
            if len(temp) == 4:
                recField, evokedAll, score, eps = eb.load.unpickle(
                    f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{thisLambdaInd}{pickleSuffix}"
                )
                del temp
            elif len(temp) == 6:
                recField, evokedAll, score, eps, regressorSTDs, epochsSTDs = (
                    eb.load.unpickle(
                        f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{thisLambdaInd}{pickleSuffix}"
                    )
                )
                del temp

            maxScores["MEG"][thisLambdaInd] = score[
                :157
            ].max()  # This will take the r value from the MEG sensor with the greatest r value
            maxScores["EEG"][thisLambdaInd] = score[
                157:
            ].max()  # This will take the r value from the EEG sensor with the greatest r value
            # maxScores[thisLambdaInd] = np.percentile(score, 90)  # This will take the r value at the 90th percentile

            if cropEvoked is not None:
                trfHere = (
                    evokedAll[regressorInd]
                    .crop(tmin=cropEvoked[0], tmax=cropEvoked[1])
                    .get_data()
                )
            else:
                trfHere = evokedAll[regressorInd].get_data()

            trfsAllLambdas.append(trfHere)

        lambdaIndToUse = int(
            np.array([maxScores["MEG"].argmax(), maxScores["EEG"].argmax()]).max()
        )  # and this will take the max index of the lambda, for this subject and this condition, which had the greatest max sensor r values for both MEG and EEG
        lambdaInfo = {
            "lambdaIndToUse": lambdaIndToUse,
            "maxScoresAllLambdas": maxScores,
            "trfsAllLambdas": trfsAllLambdas,
        }

    else:

        lambdaIndToUse = lambdaInd
        lambdaInfo = {
            "lambdaIndToUse": lambdaIndToUse,
            "maxScoresAllLambdas": {"MEG": None, "EEG": None},
            "trfsAllLambdas": None,
        }

    temp = eb.load.unpickle(
        f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaIndToUse}{pickleSuffix}"
    )
    if len(temp) == 4:
        recField, evokedAll, score, eps = eb.load.unpickle(
            f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaIndToUse}{pickleSuffix}"
        )
        del temp

    elif len(temp) == 6:
        recField, evokedAll, score, eps, regressorSTDs, epochsSTDs = eb.load.unpickle(
            f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaIndToUse}{pickleSuffix}"
        )

    evoked = evokedAll[regressorInd]

    if shiftEvoked is not None:
        tempMat = np.zeros(evoked.get_data().shape)
        fs = evoked.info["sfreq"]
        if int(shiftEvoked * fs) < 0:
            tempMat[:, : int(shiftEvoked * fs)] = evoked.get_data()[
                :, -int(shiftEvoked * fs) :
            ]
            evoked = mne.EvokedArray(
                data=tempMat, info=evoked.info, tmin=evoked.times[0]
            )
        elif int(shiftEvoked * fs) > 0:
            tempMat[:, int(shiftEvoked * fs) :] = evoked.get_data()[
                :, : -int(shiftEvoked * fs)
            ]
            evoked = mne.EvokedArray(
                data=tempMat, info=evoked.info, tmin=evoked.times[0]
            )

    # if resetAllBads:
    #     evoked.info["bads"] = []
    #     evoked.interpolate_bads()

    if len(temp) == 6:
        if normalizeByDataSTD:
            data = evoked.get_data().T
            # data = data * epochsSTDs.mean(axis=0) * lambdas[lambdaIndToUse]#/ regressorSTDs.mean()
            data = data * epochsSTDs.mean(axis=0) * multFactor  # / regressorSTDs.mean()
            evoked = mne.EvokedArray(
                data=data.T, info=evoked.info, tmin=evoked.times[0]
            )
        del temp

    # print("\n")
    # print(f"Subject {subject}")
    # print(f"Condition {filePrefix}_{typeOfRegressor}{filenameSuffix}")
    # print(f"Best lambda is number {lambdaIndToUse}")
    # print(f"Max correlation is {score.max()}")
    # print(f"Sensor of max correlation is {evoked.ch_names[score.argmax()]}")
    # print("\n")

    if useDefaultMontage:
        evoked.set_montage(mne.channels.make_standard_montage("biosemi32"))

    # # Temp version
    # recField, score, eps = eb.load.unpickle(f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle")
    # tempEvoked = mne.read_evokeds(
    #     f"{megLocation}{filePrefix}{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}-ave.fif"
    # )[0]

    # fs = tempEvoked.info["sfreq"]
    # tempMat = recField.coef_[:, regressorInd, int(edgePad * fs) : int(-edgePad * fs - 1)]

    # evoked = mne.EvokedArray(tempMat, tempEvoked.info, tmin=tempEvoked.times[0])

    if filterTRFs:
        evoked.filter(l_freq=cutoffFreq[0], h_freq=cutoffFreq[1])

    if cropEvoked is not None:
        evoked.crop(tmin=cropEvoked[0], tmax=cropEvoked[1])

    return evoked, score, lambdaInfo


def makeAnalytic(mat):
    analyticMat = sp.signal.hilbert(mat, axis=0)
    return analyticMat


def constructComplexList(
    rampFunction,
    avgMat,
    evoked,
    scores,
    comment,
    complexSubtypeNames,
    allLambdaInds,
    maxScoresAll,
    resetAllBads,
    allMaxScoresAllLambdas,
    allTrfsAllLambdas,
    zScoring,
):

    # if resetAllBads:
    #     evoked.info["bads"] = []

    if zScoring:
        avgMat[:, :157, ...] = sp.stats.zscore(
            avgMat[:, :157, ...], axis=None, nan_policy="omit"
        )
        avgMat[:, 157:, ...] = sp.stats.zscore(
            avgMat[:, 157:, ...], axis=None, nan_policy="omit"
        )

    evokedAvgList = []
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.real(np.nanmean(makeAnalytic(avgMat), axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[0],
        )
    )
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.imag(np.nanmean(makeAnalytic(avgMat), axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[1],
        )
    )
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.abs(np.nanmean(makeAnalytic(avgMat), axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[2],
        )
    )
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.angle(np.nanmean(makeAnalytic(avgMat), axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[3],
        )
    )

    realList = []
    imagList = []
    absList = []
    angleList = []
    for i in range(avgMat.shape[2]):
        realList.append(
            mne.EvokedArray(
                rampFunction * np.real(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[0],
            )
        )
        imagList.append(
            mne.EvokedArray(
                rampFunction * np.imag(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[1],
            )
        )
        absList.append(
            mne.EvokedArray(
                rampFunction * np.abs(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[2],
            )
        )
        angleList.append(
            mne.EvokedArray(
                rampFunction * np.angle(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[3],
            )
        )

    evokedList = [realList, imagList, absList, angleList]

    if resetAllBads:
        for i in evokedAvgList:
            i.info["bads"] = []

    return (
        evokedAvgList,
        evokedList,
        scores,
        allLambdaInds,
        maxScoresAll,
        allMaxScoresAllLambdas,
        allTrfsAllLambdas,
    )


def loadAndPrepareEvokeds(
    typeOfRegressor,
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
    multFactor,
    cropEvoked=None,
):

    avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

    scores = np.zeros(nChannels)

    allLambdaInds = np.zeros(len(subjectsToAverage))
    allMaxScoresAllLambdas = {"MEG": [], "EEG": []}
    maxScoresAll = np.zeros(len(subjectsToAverage))
    allTrfsAllLambdas = []

    for i, subject in enumerate(subjectsToAverage):
        megLocation = parentDir + subject + subDirs

        if not isinstance(filenameSuffix, list):
            filenameSuffix = [filenameSuffix]

        evokedDataList = []
        scoreList = []
        lambdaIndList = []
        maxScoresListMEG = []
        maxScoresListEEG = []
        allTrfsList = []
        lenToIter = len(
            filenameSuffix
        )  # This will almost always be 1, unless averaging over multiple conditions with different filename suffixes
        for suffixInd in range(lenToIter):
            if shiftEvoked is not None and shiftEvoked == "waveV":
                condChoices = [
                    "T: Quiet",
                    "T: All",
                    "D: All",
                    "M: Easy",
                    "M: Hard",
                    "T: Clicks complete",
                ]
                timesFive = eb.load.unpickle(parentDir + subject + "/timesFive.pickle")
                # timesFive = eb.load.unpickle(parentDir + subject + "/timesFiveUpdated.pickle")

                # Index the timesFive list for the correct condition, and index that array with 0 which is the first peak time, wave V
                if comment == "M: All":
                    if filenameSuffix[suffixInd][-1] == "y":  # mix_easy
                        shiftEvokedHere = 0.008 - timesFive[3][0]
                    elif filenameSuffix[suffixInd][-1] == "d":  # mix_hard
                        shiftEvokedHere = 0.008 - timesFive[4][0]
                    elif filenameSuffix[suffixInd][-1] == "B":  # mix_0dB
                        shiftEvokedHere = 0.008 - np.mean(
                            (timesFive[3][0], timesFive[4][0])
                        )  # Take the mean of mix easy and mix hard to approximate the wave V for 0 dB, because we don't actually have it measured
                else:
                    shiftEvokedHere = 0.008 - timesFive[condChoices.index(comment)][0]
            else:
                shiftEvokedHere = shiftEvoked

            evoked, score, lambdaInfo = readRecField(
                parentDir,
                subject,
                subDirs,
                filePrefix,
                typeOfRegressor,
                filenameSuffix[suffixInd],
                lambdas,
                lambdaInd,
                optimizeLambda,
                nameOfRegressor,
                nameOfRegressors,
                pickleSuffix,
                shiftEvokedHere,
                normalizeByDataSTD,
                useDefaultMontage,
                filterTRFs,
                cutoffFreq,
                timeToAddToStart,
                multFactor,
                cropEvoked,
            )
            evokedDataList.append(evoked.get_data())
            scoreList.append(score)
            lambdaIndList.append(lambdaInfo["lambdaIndToUse"])
            maxScoresListMEG.append(lambdaInfo["maxScoresAllLambdas"]["MEG"])
            maxScoresListEEG.append(lambdaInfo["maxScoresAllLambdas"]["EEG"])
            allTrfsList.append(lambdaInfo["trfsAllLambdas"])

        if lenToIter > 1:
            dataTemp = np.array(evokedDataList).mean(axis=0)
            evoked = mne.EvokedArray(
                data=dataTemp, info=evoked.info, tmin=evoked.times[0]
            )
            score = np.array(scoreList).mean(axis=0)
            lambdaInfo["lambdaIndToUse"] = int(
                np.round(np.array(lambdaIndList).mean(axis=0))
            )
            if optimizeLambda:
                lambdaInfo["maxScoresAllLambdas"]["MEG"] = np.array(
                    maxScoresListMEG
                ).mean(axis=0)
                lambdaInfo["maxScoresAllLambdas"]["EEG"] = np.array(
                    maxScoresListEEG
                ).mean(axis=0)
                lambdaInfo["trfsAllLambdas"] = np.array(allTrfsList).mean(axis=0)
            else:
                lambdaInfo["maxScoresAllLambdas"]["MEG"] = None
                lambdaInfo["maxScoresAllLambdas"]["EEG"] = None
                lambdaInfo["trfsAllLambdas"] = None

        if additionalBadChans is not None and additionalBadChans[i] is not None:
            print(f"For subject {subject}")
            print(f"Adding bad channels {additionalBadChans[i]}")
            print("\n")
            evoked.info["bads"].extend(additionalBadChans[i])

        badChanInds = [
            i
            for i, w in enumerate(evoked.ch_names)
            if (w in evoked.info["bads"]) & (w[:2] != "ME")
        ]
        evokedMat = evoked.get_data().T[
            int(timeToAddToStart * evoked.info["sfreq"]) :, :
        ]

        if nanOutBads:
            evokedMat[:, badChanInds] = np.nan
        # if zScoring:
        #     evokedMat = sp.stats.zscore(evokedMat, axis=None)

        avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
        maxScoresAll[i] = score.max()
        scores += score
        allLambdaInds[i] = lambdaInfo["lambdaIndToUse"]
        allMaxScoresAllLambdas["MEG"].append(lambdaInfo["maxScoresAllLambdas"]["MEG"])
        allMaxScoresAllLambdas["EEG"].append(lambdaInfo["maxScoresAllLambdas"]["EEG"])
        allTrfsAllLambdas.append(lambdaInfo["trfsAllLambdas"])

    scores /= i + 1
    evokedAvg = constructComplexList(
        rampFunction,
        avgMat,
        evoked,
        scores,
        comment=comment,
        complexSubtypeNames=complexSubtypeNames,
        allLambdaInds=allLambdaInds,
        maxScoresAll=maxScoresAll,
        resetAllBads=resetAllBads,
        allMaxScoresAllLambdas=allMaxScoresAllLambdas,
        allTrfsAllLambdas=allTrfsAllLambdas,
        zScoring=zScoring,
    )

    return evokedAvg


def createPeakTimes(
    avgOrAll,
    condsToPlot,
    peakRanges,
    numPeaksPerRange,
    useEEGonly,
    useMEGonly,
    order,
    plotTimes,
    eegOnly,
    eegSigns,
):

    evokedImag = [
        cond[avgOrAll][1] for cond in condsToPlot
    ]  # Get all the TRF phases, which already have been calculated
    evokedAmps = [
        cond[avgOrAll][2] for cond in condsToPlot
    ]  # Get all the TRF phases, which already have been calculated
    evokedAngles = [
        cond[avgOrAll][3] for cond in condsToPlot
    ]  # Get all the TRF phases, which already have been calculated

    normalizeHilbert = True

    times = []
    for evInd, evoked in enumerate(evokedAngles):
        timesHere = []
        for i in range(len(peakRanges)):
            # This gfp etc. code here is adapted from MNE internal code, because there aren't ways to customize the "peaks" option
            evokedTemp = evoked.copy().crop(
                tmin=peakRanges[i][0], tmax=peakRanges[i][1]
            )
            picksEEG = mne.pick_types(
                evokedTemp.info, meg=False, eeg=True, exclude="bads"
            )
            modulator = (
                evokedAmps[evInd]
                .copy()
                .crop(tmin=peakRanges[i][0], tmax=peakRanges[i][1])
                .get_data(picks=picksEEG)
            )
            phasesForSign = np.cos(evokedTemp.get_data(picks=picksEEG))
            phaseHist = np.cos(evokedTemp.get_data(picks=picksEEG)) ** 2
            if normalizeHilbert:
                modulator = modulator / np.max(np.abs(modulator))
                phaseHist = phaseHist / np.max(np.abs(phaseHist))
            gfp = (modulator * phaseHist).mean(axis=0)
            gfpForSign = (modulator * phasesForSign).mean(axis=0)
            if normalizeHilbert:
                gfp = gfp / np.max(np.abs(gfp))
            # gfp = sp.ndimage.gaussian_filter1d(gfp, 4)
            peaks = sp.signal.argrelmax(gfp, order=order, axis=0)[0]
            if len(peaks) > numPeaksPerRange[i]:
                max_indices = np.argsort(
                    gfp[peaks]
                )  # This gives the peak location indices, sorted from smallest to biggest peak
                if eegSigns[i] is not None:
                    peaks = np.sort(
                        peaks[max_indices]
                    )  # This sorts the peak locations from earliest to latest
                    j = 0
                    peakInd = 0
                    peaksFinal = []
                    while j < len(eegSigns[i]):
                        if eegSigns[i][j] == "neg":
                            if gfpForSign[peaks[peakInd]] < 0:
                                peaksFinal.append(peaks[peakInd])
                                j += 1
                        elif eegSigns[i][j] == "pos":
                            if gfpForSign[peaks[peakInd]] > 0:
                                peaksFinal.append(peaks[j])
                                j += 1
                        peakInd += 1
                    peaks = np.array(peaksFinal)
                else:
                    peaks = np.sort(
                        peaks[max_indices[-numPeaksPerRange[i] :]]
                    )  # This sorts the peak locations from earliest to latest
            # else:
            #     print(f"Only one peak found")

            tempEEG = evokedTemp.times[peaks]

            if not eegOnly:
                picksMEG = mne.pick_types(
                    evokedTemp.info, meg=True, eeg=False, exclude="bads"
                )
                modulator = (
                    evokedAmps[evInd]
                    .copy()
                    .crop(tmin=peakRanges[i][0], tmax=peakRanges[i][1])
                    .get_data(picks=picksMEG)
                )
                phaseHist = np.cos(evokedTemp.get_data(picks=picksMEG)) ** 2
                if normalizeHilbert:
                    modulator = modulator / np.max(np.abs(modulator))
                    phaseHist = phaseHist / np.max(np.abs(phaseHist))
                gfp = (modulator * phaseHist).mean(axis=0)
                if normalizeHilbert:
                    gfp = gfp / np.max(np.abs(gfp))
                # gfp = sp.ndimage.gaussian_filter1d(gfp, 4)
                peaks = sp.signal.argrelmax(gfp, order=order, axis=0)[0]
                if len(peaks) > numPeaksPerRange[i]:
                    max_indices = np.argsort(gfp[peaks])[-numPeaksPerRange[i] :]
                    peaks = np.sort(peaks[max_indices])
                tempMEG = evokedTemp.times[peaks]
                if len(tempMEG) < len(tempEEG):
                    newMEG = np.zeros(tempEEG.shape)
                    for j in range(len(tempMEG)):
                        thisMEGtime = tempMEG[j]
                        thisInd = np.argmin(np.abs(thisMEGtime - tempEEG))
                        newMEG[thisInd] = thisMEGtime
                    tempMEG = newMEG
                temp = []
                for j in range(len(tempEEG)):
                    # for j in range(len(tempMEG)):
                    if useEEGonly[i][j]:
                        temp.append(tempEEG[j])
                    else:
                        if tempMEG[j] != 0:
                            if useMEGonly[i][j]:
                                temp.append(tempMEG[j])
                            else:
                                temp.append(np.mean((tempEEG[j], tempMEG[j])))
                # for j in range(np.min((len(tempEEG), len(tempMEG)))):
                for j in range(len(temp)):
                    timesHere.append(np.array(temp[j]))
            else:
                for j in range(len(tempEEG)):
                    timesHere.append(tempEEG[j])
        times.append(np.array(timesHere))

        picksEEG = mne.pick_types(evoked.info, meg=False, eeg=True, exclude="bads")
        modulatorAllEEG = evokedAmps[evInd].get_data(picks=picksEEG)
        phaseHistAllEEG = np.cos(evoked.get_data(picks=picksEEG)) ** 2
        if normalizeHilbert:
            modulatorAllEEG = modulatorAllEEG / np.max(np.abs(modulatorAllEEG))
            phaseHistAllEEG = phaseHistAllEEG / np.max(np.abs(phaseHistAllEEG))
        gfpAllEEG = modulatorAllEEG * phaseHistAllEEG
        if normalizeHilbert:
            gfpAllEEG = gfpAllEEG / np.max(np.abs(gfpAllEEG))

        if not eegOnly:
            picksMEG = mne.pick_types(evoked.info, meg=True, eeg=False, exclude="bads")
            modulatorAllMEG = evokedAmps[evInd].get_data(picks=picksMEG)
            phaseHistAllMEG = np.cos(evoked.get_data(picks=picksMEG)) ** 2
            if normalizeHilbert:
                modulatorAllMEG = modulatorAllMEG / np.max(np.abs(modulatorAllMEG))
                phaseHistAllMEG = phaseHistAllMEG / np.max(np.abs(phaseHistAllMEG))
            gfpAllMEG = modulatorAllMEG * phaseHistAllMEG
            if normalizeHilbert:
                gfpAllMEG = gfpAllMEG / np.max(np.abs(gfpAllMEG))

        if plotTimes:
            modMeanEEG = modulatorAllEEG.mean(axis=0) / np.max(
                np.abs(modulatorAllEEG.mean(axis=0))
            )
            phaseMeanEEG = phaseHistAllEEG.mean(axis=0) / np.max(
                np.abs(phaseHistAllEEG.mean(axis=0))
            )
            gfpMeanEEG = gfpAllEEG.mean(axis=0) / np.max(np.abs(gfpAllEEG.mean(axis=0)))
            if eegOnly:
                plt.figure()
                plt.plot(
                    evoked.times,
                    modMeanEEG,
                    evoked.times,
                    phaseMeanEEG,
                    evoked.times,
                    gfpMeanEEG,
                )
                for i in timesHere:
                    plt.axvline(i, color="black")
                plt.title(f"EEG {evoked.comment}")
                plt.legend(
                    [
                        "Mean Hilbert amplitude",
                        "Cosine squared phases",
                        "Phases modulated by amplitudes",
                    ]
                )
            else:
                modMeanMEG = modulatorAllMEG.mean(axis=0) / np.max(
                    np.abs(modulatorAllMEG.mean(axis=0))
                )
                phaseMeanMEG = phaseHistAllMEG.mean(axis=0) / np.max(
                    np.abs(phaseHistAllMEG.mean(axis=0))
                )
                gfpMeanMEG = gfpAllMEG.mean(axis=0) / np.max(
                    np.abs(gfpAllMEG.mean(axis=0))
                )

                figPhase, axPhase = plt.subplots(2, 1, figsize=[20, 10])
                axPhase[0].plot(
                    evoked.times,
                    modMeanEEG,
                    evoked.times,
                    phaseMeanEEG,
                    evoked.times,
                    gfpMeanEEG,
                )
                for i in timesHere:
                    axPhase[0].axvline(i, color="black")
                axPhase[0].set_title(f"EEG {evoked.comment}")
                # plt.legend(["Mean Hilbert amplitude", "Cosine squared phases", "Phases modulated by amplitudes"])

                axPhase[1].plot(
                    evoked.times,
                    modMeanMEG,
                    evoked.times,
                    phaseMeanMEG,
                    evoked.times,
                    gfpMeanMEG,
                )
                plt.legend(
                    [
                        "Mean Hilbert amplitude",
                        "Cosine squared phases",
                        "Phases modulated by amplitudes",
                    ]
                )
                for i in timesHere:
                    axPhase[1].axvline(i, color="black")
                axPhase[1].set_title(f"MEG {evoked.comment}")

    return times
