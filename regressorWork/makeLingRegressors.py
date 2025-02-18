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
import numpy as np
import scipy as sp
import eelbrain as eb
import textgrid as tg
import matplotlib
import matplotlib.pyplot as plt

from neuroAndSignalTools.freqAnalysis import *

matplotlib.use("QtAgg")
plt.ion()

# %%
# These are the four master condition and distractor orders
#  Use them TOGETHER, i.e. only uncomment the first ones together
#  (condition and distractor orders), or the third ones together, etc.
# conditionOrder=[6,13,10,3,1,8,14,7,12,9,4,15,5,11,2,16];prependText="A"
# conditionOrder=[7,5,16,2,13,6,11,1,15,4,9,10,8,14,3,12];prependText="B"
# conditionOrder=[4,9,10,8,1,3,14,7,12,13,2,11,6,15,5,16];prependText="C"
# conditionOrder=[8,5,15,4,9,10,13,2,11,6,16,1,7,14,3,12];prependText="D"


# distractorOrder=[7,1,6,5,1,11,9,2,1,1,12,4,1,3,10,8];
# distractorOrder=[10,1,11,2,1,6,4,1,12,8,1,7,1,9,3,5];
# distractorOrder=[1,1,6,10,1,5,4,12,2,1,8,7,11,9,1,3];
# distractorOrder=[5,1,6,4,1,8,1,9,1,4,11,1,10,7,2,12];

textgridMainDir = "/Users/karl/Dropbox/UMD/audioMixes/mfaWork/outputs/"
stimMainDir = "/Volumes/Seagate/map/stimAndPredictors/"


# This block of numbers is copied from MATLAB and so is one-indexed
# Therefore just subtract one to use them as indices in Python
conditionOrders = []
conditionOrders.append([6, 13, 10, 3, 1, 8, 14, 7, 12, 9, 4, 15, 5, 11, 2, 16])
conditionOrders.append([7, 5, 16, 2, 13, 6, 11, 1, 15, 4, 9, 10, 8, 14, 3, 12])
conditionOrders.append([4, 9, 10, 8, 1, 3, 14, 7, 12, 13, 2, 11, 6, 15, 5, 16])
conditionOrders.append([8, 5, 15, 4, 9, 10, 13, 2, 11, 6, 16, 1, 7, 14, 3, 12])
conditionOrders = np.array(conditionOrders) - 1

distractorOrders = []
distractorOrders.append([7, 1, 6, 5, 1, 11, 9, 2, 1, 1, 12, 4, 1, 3, 10, 8])
distractorOrders.append([10, 1, 11, 2, 1, 6, 4, 1, 12, 8, 1, 7, 1, 9, 3, 5])
distractorOrders.append([1, 1, 6, 10, 1, 5, 4, 12, 2, 1, 8, 7, 11, 9, 1, 3])
distractorOrders.append([5, 1, 6, 4, 1, 8, 1, 9, 1, 4, 11, 1, 10, 7, 2, 12])
distractorOrders = np.array(distractorOrders) - 1


skipDist = np.array(
    [[2, 5, 10, 13], [2, 5, 8, 11], [2, 5, 10, 15], [2, 5, 7, 12]]
)  # One-indexed indices! Skip these distractors, because these trials have no distractor
# We'll use these below as normal Python indices, so just subtract 1 from the one-indexed numbers from MATLAB
skipDist = skipDist - 1


prependText = []
prependText.append("A")
prependText.append("B")
prependText.append("C")
prependText.append("D")

faTargPref = "american/female/"
fbTargPref = "british/female/"
maTargPref = "american/male/"
mbTargPref = "british/male/"

faDistPref = "american/female/dist-"
fbDistPref = "british/female/dist-"
maDistPref = "american/male/dist-"
mbDistPref = "british/male/dist-"

matrixOrder = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

targets = [
    faTargPref,
    fbTargPref,
    maTargPref,
    mbTargPref,
    faTargPref,
    fbTargPref,
    faTargPref,
    fbTargPref,
    maTargPref,
    mbTargPref,
    maTargPref,
    mbTargPref,
    faTargPref,
    maTargPref,
    faTargPref,
    maTargPref,
]
targets = [targets[i] for i in matrixOrder]

distractors = [
    faDistPref,
    faDistPref,
    faDistPref,
    faDistPref,  # Remember these four (MATLAB 1,5,9,13, or Python 0,4,8,12) don't matter, just put something
    fbDistPref,
    faDistPref,
    fbDistPref,
    faDistPref,
    mbDistPref,
    maDistPref,
    mbDistPref,
    maDistPref,
    maDistPref,
    faDistPref,
    maDistPref,
    faDistPref,
]
distractors = [distractors[i] for i in matrixOrder]

fs = 500  # Sampling rate of regressors to make; probably 2000, which would be the same as other slower time-scale regressors such as envelope etc.


doGaussian = True
# gaussianSD = 10 * fs / 1000  # One standard deviation of the gaussians that will be made at the onset times, in units of samples (ms * fs / 1000)
gaussianSD = (
    15 * fs / 1000
)  # One standard deviation of the gaussians that will be made at the onset times, in units of samples (ms * fs / 1000)
# gaussianSD = 20 * fs / 1000  # One standard deviation of the gaussians that will be made at the onset times, in units of samples (ms * fs / 1000)

# wordPredictorName = "~wordOnsets.pickle"
# phonePredictorName = "~phoneOnsets.pickle"
# wordPredictorName = "~wordOnsets_gaussian.pickle"
# phonePredictorName = "~phoneOnsets_gaussian.pickle"
# wordPredictorName = "~wordOnsets_gaussian20SD.pickle"
# phonePredictorName = "~phoneOnsets_gaussian20SD.pickle"
# wordPredictorName = "~wordOnsets_gaussian40SD.pickle"
# phonePredictorName = "~phoneOnsets_gaussian40SD.pickle"
wordPredictorName = "~wordOnsets_gaussian15msSD.pickle"
phonePredictorName = "~phoneOnsets_gaussian15msSD.pickle"

# %%
for counterbalance in range(
    len(conditionOrders)
):  # This is 4 counterbalances, but just write it out to make it explicit
    for i in range(
        len(conditionOrders[counterbalance])
    ):  # This is always 16 trials/conditions, but just write it out to make it explicit

        # Index both of these lists with the conditionOrders matrix, because the entries in the targets and distractors lists correspond to those numbers
        target = targets[conditionOrders[counterbalance, i]]
        distractor = distractors[conditionOrders[counterbalance, i]]

        targetTextgrid = tg.TextGrid.fromFile(
            textgridMainDir + target + str(i).zfill(2) + ".TextGrid"
        )  # Read in the appropriate speaker's target text, which always come in order
        distractorTextgrid = tg.TextGrid.fromFile(
            textgridMainDir
            + distractor
            + str(distractorOrders[counterbalance, i]).zfill(2)
            + ".TextGrid"
        )  # and read in the appropriate distractor text by indexing into the distractorOrders matrix

        targetWords = np.zeros(len(targetTextgrid[0]))
        targetPhones = np.zeros(len(targetTextgrid[1]))
        distractorWords = np.zeros(len(distractorTextgrid[0]))
        distractorPhones = np.zeros(len(distractorTextgrid[1]))

        for wordInd in range(len(targetWords)):
            targetWords[wordInd] = targetTextgrid[0][wordInd].minTime

        for phoneInd in range(len(targetPhones)):
            targetPhones[phoneInd] = targetTextgrid[1][phoneInd].minTime

        if not np.any(
            i == skipDist[counterbalance]
        ):  # If we're on a trial/condition with no distractor, then just leave the vectors of onsets as all zeros
            for wordInd in range(len(distractorWords)):
                distractorWords[wordInd] = distractorTextgrid[0][wordInd].minTime

            for phoneInd in range(len(distractorPhones)):
                distractorPhones[phoneInd] = distractorTextgrid[1][phoneInd].minTime

        # Get the mixture audio to know exactly how long to make the regressor time series
        fsAudio, stimulus = sp.io.wavfile.read(
            f"{stimMainDir}mixes/{prependText[counterbalance]}_mix{i+1}.wav"
        )  # Index with i+1 because the filenames have one-indexed indices

        # Index all four of these with 1:-2 to exclude the first and last, because typically those are just None from MFA, not actually a word or phone, for whatever reason
        targetWordPredictor = createPredictorTimeseries(
            targetWords[1:-2], fs, int(len(stimulus) * fs / fsAudio), 1
        )
        targetPhonePredictor = createPredictorTimeseries(
            targetPhones[1:-2], fs, int(len(stimulus) * fs / fsAudio), 1
        )

        if doGaussian:
            targetWordPredictor = sp.ndimage.gaussian_filter1d(
                targetWordPredictor, gaussianSD
            )
            targetPhonePredictor = sp.ndimage.gaussian_filter1d(
                targetPhonePredictor, gaussianSD
            )

        # This won't work for the distractors that we cut off to be shorter, because the text was still analyzed by MFA
        # so just cut off all onsets greater than the length of the mixture audio that has been read in
        distractorWords = distractorWords[distractorWords < stimulus.shape[0] / fsAudio]
        distractorPhones = distractorPhones[
            distractorPhones < stimulus.shape[0] / fsAudio
        ]

        distractorWordPredictor = createPredictorTimeseries(
            distractorWords[1:-2], fs, int(len(stimulus) * fs / fsAudio), 1
        )
        distractorPhonePredictor = createPredictorTimeseries(
            distractorPhones[1:-2], fs, int(len(stimulus) * fs / fsAudio), 1
        )

        if doGaussian:
            distractorWordPredictor = sp.ndimage.gaussian_filter1d(
                distractorWordPredictor, gaussianSD
            )
            distractorPhonePredictor = sp.ndimage.gaussian_filter1d(
                distractorPhonePredictor, gaussianSD
            )

        eb.save.pickle(
            targetWordPredictor,
            f"{stimMainDir}targets/predictors/{prependText[counterbalance]}_target{i+1}{wordPredictorName}",
        )
        eb.save.pickle(
            targetPhonePredictor,
            f"{stimMainDir}targets/predictors/{prependText[counterbalance]}_target{i+1}{phonePredictorName}",
        )

        if not np.any(
            i == skipDist[counterbalance]
        ):  # If we're on a trial/condition with no distractor, simply don't write distractor linguistic predictors, as they would just be all zeros
            eb.save.pickle(
                distractorWordPredictor,
                f"{stimMainDir}distractors/predictors/{prependText[counterbalance]}_distractor{i+1}{wordPredictorName}",
            )
            eb.save.pickle(
                distractorPhonePredictor,
                f"{stimMainDir}distractors/predictors/{prependText[counterbalance]}_distractor{i+1}{phonePredictorName}",
            )

        eb.save.pickle(
            targetWordPredictor + distractorWordPredictor,
            f"{stimMainDir}mixes/predictors/{prependText[counterbalance]}_mix{i+1}{wordPredictorName}",
        )
        eb.save.pickle(
            targetPhonePredictor + distractorPhonePredictor,
            f"{stimMainDir}mixes/predictors/{prependText[counterbalance]}_mix{i+1}{phonePredictorName}",
        )

        print(
            f"Done creating and writing counterbalance {counterbalance} and trial {i+1}"
        )
        print("\n")
