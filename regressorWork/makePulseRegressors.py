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

allTargetFilenames = sorted(glob.glob(f"{parentDir}targets/*target*wav"))
allDistractorFilenames = sorted(glob.glob(f"{parentDir}distractors/*distractor*wav"))

skipDist = np.array(
    [[2, 5, 10, 13], [2, 5, 8, 11], [2, 5, 10, 15], [2, 5, 7, 12]]
)  # One-indexed indices! Skip these distractors, because these trials have no distractor


conditions = ["A", "B", "C", "D"]

# %%
fs = 16384
# fs = 5000

f0_min = 40
f0_max = 600


# %%
for trial in np.arange(1, 17):
    for condInd, condition in enumerate(conditions):

        targetFilename = f"{parentDir}targets/{condition}_target{trial}.wav"

        fsAudio, targetWav = sp.io.wavfile.read(targetFilename)
        # targetWav = sp.signal.resample(targetWav, int(len(targetWav)*fs / fsAudio))
        sound = parselmouth.Sound(targetFilename)
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

        targetPulsePred = createPredictorTimeseries(
            glottal_pulse_times, fs, len(t), values=1
        )
        targetWav = sp.signal.resample(targetWav, len(targetPulsePred))
        targetHalfWave = targetWav.copy()
        targetHalfWave[targetHalfWave < 0] = 0

        if not np.any(
            trial == skipDist[condInd]  # trial and skipDist are both one-indexed
        ):  # If we're on a trial/condition with no distractor, simply don't make the predictors, and don't try to save them further down

            distractorFilename = (
                f"{parentDir}distractors/{condition}_distractor{trial}.wav"
            )

            fsAudio, distractorWav = sp.io.wavfile.read(distractorFilename)
            # targetWav = sp.signal.resample(testWav, int(len(targetWav)*fs / fsAudio))
            sound = parselmouth.Sound(distractorFilename)
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

            distractorPulsePred = createPredictorTimeseries(
                glottal_pulse_times, fs, len(t), values=1
            )

            distractorWav = sp.signal.resample(distractorWav, len(distractorPulsePred))
            distractorHalfWave = distractorWav.copy()
            distractorHalfWave[distractorHalfWave < 0] = 0
        else:
            distractorPulsePred = np.zeros(targetPulsePred.shape)

        mixPulsePred = targetPulsePred + distractorPulsePred

        mixFilename = f"{parentDir}mixes/{condition}_mix{trial}.wav"

        fsAudio, mixWav = sp.io.wavfile.read(mixFilename)

        mixWav = sp.signal.resample(mixWav, len(mixPulsePred))
        mixHalfWave = mixWav.copy()
        mixHalfWave[mixHalfWave < 0] = 0

        if fs == 16384:
            suffix = "maxFs"
        elif fs == 5000:
            suffix = "fs5000"

        savePath = f"{parentDir}targets/predictors/{condition}_target{trial}_glottalPulses_{suffix}.pickle"
        eb.save.pickle(targetPulsePred, savePath)
        savePathHalfWave = f"{parentDir}targets/predictors/{condition}_target{trial}_halfWave_{suffix}.pickle"
        eb.save.pickle(targetHalfWave, savePathHalfWave)
        print(f"Saved {savePath} and {savePathHalfWave}")
        print("\n")

        if not np.any(trial == skipDist[condInd]):
            savePath = f"{parentDir}distractors/predictors/{condition}_distractor{trial}_glottalPulses_{suffix}.pickle"
            eb.save.pickle(distractorPulsePred, savePath)
            savePathHalfWave = f"{parentDir}distractors/predictors/{condition}_distractor{trial}_halfWave_{suffix}.pickle"
            eb.save.pickle(distractorHalfWave, savePathHalfWave)
            print(f"Saved {savePath} and {savePathHalfWave}")
            print("\n")

        savePath = f"{parentDir}mixes/predictors/{condition}_mix{trial}_glottalPulses_{suffix}.pickle"
        eb.save.pickle(mixPulsePred, savePath)
        savePathHalfWave = f"{parentDir}mixes/predictors/{condition}_mix{trial}_halfWave_{suffix}.pickle"
        eb.save.pickle(mixHalfWave, savePathHalfWave)
        print(f"Saved {savePath} and {savePathHalfWave}")
        print("\n")
