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
import os
import numpy as np
import scipy as sp
import eelbrain as eb
import matplotlib.pyplot as plt
import matplotlib

from neuroAndSignalTools.freqAnalysis import *

matplotlib.use("QtAgg")
plt.ion()

# %%
iciFilename = "randomICIs.pickle"
wavFilename = "fullABRstim.wav"
toneWavfilename = "toneNew.wav"

#########################################################################
#########################################################################
#########################################################################
###########################Careful#######################################
overwrite = False
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

fs = 24000
lenClick = 1 / 5000  # 200 microseconds
clickAmp = 0.9

desiredNumTones = 200


click = np.ones(int(round(lenClick * fs))) * clickAmp

toneFs, tone = sp.io.wavfile.read(toneWavfilename)
print(f"Original tone length is {len(tone)} at sampling rate {toneFs}")
tone = sp.signal.resample(tone, int(round(len(tone) * fs / toneFs)))
tone = tone * clickAmp / np.max(np.abs(tone))
print(
    f"New tone length is {len(tone)} after resampling to sampling rate {fs}, which is {len(tone) / fs} seconds"
)

numClicks = 12000

moduloForTones = int(numClicks / desiredNumTones)
offsets = np.arange(-20, 21)

if os.path.exists(iciFilename) and not overwrite:

    allICIs, allICIsAlternate, clickIndsForTones = eb.load.unpickle(iciFilename)

else:

    clickIndsForTones = np.zeros(desiredNumTones)
    presentInd = np.copy(moduloForTones)
    for i in range(desiredNumTones):
        clickIndsForTones[i] = presentInd
        presentInd += int(moduloForTones + np.random.choice(offsets))

    allICIs = 1 / np.logspace(np.log10(5), np.log10(230), numClicks)
    np.random.shuffle(allICIs)
    allICIsAlternate = allICIs.copy()
    np.random.shuffle(allICIsAlternate)

    eb.save.pickle((allICIs, allICIsAlternate, clickIndsForTones), iciFilename)


fullTimeSeries = np.zeros(int((allICIs.sum() + 0.001) * fs))

count = 0
countAlternate = 0
countTotalTones = 0
clickTimes = []
toneTimes = []
prevICIvalues = [0]
for i in range(numClicks):
    thisClickPeriod = np.zeros(int(round(allICIs[i] * fs)))
    thisTonePeriod = np.zeros(int(round(allICIsAlternate[i] * fs)))
    # thisClick[: len(click)] = click
    fullTimeSeries[count : count + len(click)] += click
    clickTimes.append(count)  # These numbers are in sample time right now
    if i != 0:
        prevICIvalues.append(allICIs[i - 1])

    # if not i%moduloForTones and i!=0:
    if countTotalTones < desiredNumTones and i == clickIndsForTones[countTotalTones]:
        # fullTimeSeries[count:count + len(tone)] = tone
        fullTimeSeries[countAlternate : countAlternate + len(tone)] += tone
        toneTimes.append(countAlternate)  # These numbers are in sample time now
        # print(moduloForTones)
        countTotalTones += 1

    count += len(thisClickPeriod)
    countAlternate += len(thisTonePeriod)

print(f"Total tones written is {countTotalTones}")

# fullTimeSeries = fullTimeSeries * .99 / np.max(np.abs(fullTimeSeries))

t = np.linspace(0, len(fullTimeSeries) / fs, len(fullTimeSeries))

sp.io.wavfile.write(wavFilename, fs, fullTimeSeries)

# %%
print(
    f"Overall level of stim is {20 * np.log10(np.std(fullTimeSeries) / .00002)} dB SPL"
)
print(
    f"Length of stim is {len(fullTimeSeries) / fs} seconds which is {len(fullTimeSeries)} samples"
)

# %%
plt.figure()
plt.plot(t, fullTimeSeries)
# plt.plot(t[: 2 * fs], fullTimeSeries[: 2 * fs])
# plt.plot(t[-40 * fs :], fullTimeSeries[-40 * fs :])
# plt.figure()
# plt.plot(tone)

# %%
clickTimes = np.array(clickTimes) / fs
toneTimes = np.array(toneTimes) / fs
eb.save.pickle((clickTimes, prevICIvalues), "clickTimes.pickle")
eb.save.pickle(toneTimes, "toneTimes.pickle")
