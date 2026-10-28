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
import parselmouth
import numpy as np
import scipy as sp
import glob
import os
import eelbrain as eb
import matplotlib
import matplotlib.pyplot as plt

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *

matplotlib.use("QtAgg")
plt.ion()

# %%
parentDir = "/Volumes/Seagate/map/stimAndPredictors/"
praatFileDir = "/Users/karl/eegAndMegAnalysis/regressorWork/"

allTargetFilenames = sorted(glob.glob(f"{parentDir}targets/*target*wav"))
allDistractorFilenames = sorted(glob.glob(f"{parentDir}distractors/*distractor*wav"))
allMixFilenames = sorted(glob.glob(f"{parentDir}mixes/*mix*wav"))

readPulsePred = True

# %%
fs = 5000

testFilename = allMixFilenames[41]
predSubDir = "mixes"

# testFilename = allDistractorFilenames[21]
# predSubDir = "distractors"

ANmodel = eb.load.unpickle(
    f"{parentDir}{predSubDir}/predictors/{os.path.basename(testFilename)[:-4]}_ANmodel_maxFs-1.pickle"
)

# testFilename = "/Users/karl/eegAndMegAnalysis/regressorWork/clickABRstim.wav"
# ANmodel = eb.load.unpickle("/Users/karl/eegAndMegAnalysis/regressorWork/clickANmodelEeg.pickle").squeeze()


fsAudio, testWav = sp.io.wavfile.read(testFilename)
testWav = sp.signal.resample(testWav, int(len(testWav) * fs / fsAudio))
ANmodel = sp.signal.resample(
    ANmodel, len(testWav)
)  # It's fine to just use the length of the audio for resampling, because we already know the ANmodel was made from this audio file

# %%
f0_min = 40
f0_max = 600

# parselmouth.praat.run_file(f"{praatFileDir}wav_to_pitch.praat", '%s' % test[:-4], str(f0_min), str(f0_max))

# with open(test[:-4] + '.Pitch') as fid:
#     pitch = list(np.copy(fid.readlines()))[0][:-1]

# f0_min, f0_max, f_mean, f_std = [float(d) for d in pitch.split('_')]

# parselmouth.praat.run_file(f"{praatFileDir}wav_to_pulses.praat", '%s' % test[:-4], str(f0_min), str(f0_max))
# read in the pulse times

# with open(test[:-4] + '.PointProcess') as fid:
#     lines = np.copy(fid.readlines())

# assert len(lines) > 7
# pulse_times = np.array([float(l.replace(' ', '').strip().split('=')[1]) for l in lines[7:]])


sound = parselmouth.Sound(testFilename)
sound = sound.resample(fs)
t = sound.ts()
# point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, peaks)", f0_min, f0_max, "no", "yes")
# point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, peaks)", f0_min, f0_max, "yes", "no")
point_process = parselmouth.praat.call(
    sound, "To PointProcess (periodic, cc)", f0_min, f0_max
)
numPoints = parselmouth.praat.call(point_process, "Get number of points")
glottal_pulse_times = []
for i in range(1, numPoints + 1):
    glottal_pulse_times.append(
        parselmouth.praat.call(point_process, "Get time from index", i)
    )

glottal_pulse_times = np.array(glottal_pulse_times)


if readPulsePred:
    if fs == 16384:
        suffix = "maxFs"
    elif fs == 5000:
        suffix = "fs5000"
    pulse_pred_pm_read = eb.load.unpickle(
        f"{parentDir}{predSubDir}/predictors/{os.path.basename(testFilename)[:-4]}_glottalPulses_{suffix}.pickle"
    )
pulse_pred_pm = createPredictorTimeseries(glottal_pulse_times, fs, len(t), values=1)


pitch = sound.to_pitch()
pitchInterped = pitch.interpolate()
contour = pitchInterped.selected_array["frequency"]

beginCount = 0
if contour[beginCount] == 0:
    while contour[beginCount] == 0:
        beginCount += 1
        beginFreq = contour[beginCount]
    contour[0:beginCount] = beginFreq

endCount = len(contour) - 1
if contour[endCount] == 0:
    while contour[endCount] == 0:
        endCount -= 1
        endFreq = contour[endCount]
    contour[endCount + 1 :] = endFreq

# tShort = np.linspace(t[0], t[-1], len(contour))
tShort = pitch.xs()
createInterpedContour = sp.interpolate.CubicSpline(tShort, contour)
interpedContour = createInterpedContour(t)

# %%
# pulse_pred_pm /= np.max(np.abs(pulse_pred_pm))
testWav /= np.max(np.abs(testWav))
ANmodel /= np.max(np.abs(ANmodel))

# %%
# plt.plot(pulse_pred / np.max(np.abs(pulse_pred)))
fig, ax = plt.subplots(figsize=[12, 8])
ax.plot(t, pulse_pred_pm)
if readPulsePred:
    ax.plot(t, pulse_pred_pm_read)
if testWav.shape[0] == pulse_pred_pm.shape[0] - 1:
    testWav = np.append(testWav, 0)
ax.plot(t, testWav)
if ANmodel.shape[0] == pulse_pred_pm.shape[0] - 1:
    ANmodel = np.append(ANmodel, 0)
ax.plot(t, ANmodel)
ax.set_xlabel("Time (s)")
if readPulsePred:
    plt.legend(
        ["Glottal pulse times", "Glottal pulse times read", "Audio", "AN model"],
        loc="upper left",
    )
else:
    plt.legend(["Glottal pulse times", "Audio", "AN model"], loc="upper left")
# ax2 = ax.twinx()
# ax2.plot(t, interpedContour, color="red")
# interpedContourDeriv = np.diff(interpedContour, prepend=interpedContour[0])
# interpedContourDeriv = interpedContourDeriv / np.max(np.abs(interpedContourDeriv)) * np.max(np.abs(interpedContour))
# ax2.plot(t, interpedContourDeriv, color="purple")
# ax2.set_ylabel("Frequency (Hz)")
# plt.legend(["Fundamental frequency (F0)", "F0 derivative (unitless)"], loc="upper right")

# %%
testWav = sp.stats.zscore(testWav)
ANmodel = sp.stats.zscore(ANmodel)

# %%
norm_factor = np.sqrt(np.sum(np.abs(testWav) ** 2) * np.sum(np.abs(ANmodel) ** 2))
crossCorr = sp.signal.correlate(ANmodel, testWav, mode="full") / norm_factor
lags = sp.signal.correlation_lags(len(ANmodel), len(testWav), mode="full")
lags = lags * 1 / fs * 1000  # Now the lags are in milliseconds

corrPlotTimes = [-15, 32]  # Start and end of plot in ms
corrPlotInds = []
corrPlotInds.append(np.abs(corrPlotTimes[0] - lags).argmin())
corrPlotInds.append(np.abs(corrPlotTimes[1] - lags).argmin())
maxTime = lags[np.argmax(crossCorr)]
maxCorr = crossCorr[np.argmax(crossCorr)]

figC, axC = plt.subplots(figsize=[14, 9])
axC.plot(
    lags[corrPlotInds[0] : corrPlotInds[1]],
    crossCorr[corrPlotInds[0] : corrPlotInds[1]],
    linewidth=3.5,
)
axC.axvline(x=maxTime, color="black", linewidth=2)
axC.set_xlabel("Lags (ms)")
axC.set_ylabel("Correlation coefficient")
axC.set_title(f"Max correlation of {maxCorr} at {maxTime} ms")

# %%
