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
import matplotlib
import matplotlib.pyplot as plt
import glob
import cochlea
import time
import joblib

from neuroAndSignalTools.freqAnalysis import *

matplotlib.use("QtAgg")
plt.ion()

# %load_ext autoreload
# %autoreload 2

# %%
doParallel = True
n_jobs = 16
parallelBackend = "loky"

# regressorDir='/Users/karl/Dropbox/UMD/ffrTests/'

# mainDirs=['/Users/karl/map/stimAndPredictors/targets/','/Users/karl/map/stimAndPredictors/distractors/','/Users/karl/map/stimAndPredictors/mixes/']
# regTypes=['target','distractor','mix']

mainDirSuff = ""
# mainDirSuff = "Inverted"

mainDirs = [
    f"/Volumes/Seagate/map/stimAndPredictors/targets{mainDirSuff}/",
    f"/Volumes/Seagate/map/stimAndPredictors/distractors{mainDirSuff}/",
    f"/Volumes/Seagate/map/stimAndPredictors/mixes{mainDirSuff}/",
]
regTypes = ["target", "distractor", "mix"]
conditions = ["A", "B", "C", "D"]

# mainDirs=['/Users/karl/map/stimAndPredictors/mixes/']
# regTypes=['mix']
# conditions=['D']
skipDist = np.array(
    [[2, 5, 10, 13], [2, 5, 8, 11], [2, 5, 10, 15], [2, 5, 7, 12]]
)  # One-indexed indices! Skip these distractors, because these trials have no distractor

# dBSPL=70

# These are the approximate levels of the targets and distractors separately, such that their sum is always 70 dBSPL, copied and pasted from the MATLAB script that generates the stimulus wav files
dBSPLs = np.stack(
    (
        np.array(
            [
                [
                    68.5000000000000,
                    70,
                    67,
                    68.5000000000000,
                    70,
                    67,
                    67,
                    68.5000000000000,
                    64.5000000000000,
                    70,
                    67,
                    67,
                    70,
                    67,
                    68.5000000000000,
                    64.5000000000000,
                ],
                [
                    64.5000000000000,
                    0,
                    67,
                    64.5000000000000,
                    0,
                    67,
                    67,
                    64.5000000000000,
                    68.5000000000000,
                    0,
                    67,
                    67,
                    0,
                    67,
                    64.5000000000000,
                    68.5000000000000,
                ],
                [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70],
            ]
        ),
        np.array(
            [
                [
                    68.5000000000000,
                    70,
                    64.5000000000000,
                    68.5000000000000,
                    70,
                    68.5000000000000,
                    67,
                    70,
                    67,
                    67,
                    70,
                    67,
                    67,
                    67,
                    68.5000000000000,
                    64.5000000000000,
                ],
                [
                    64.5000000000000,
                    0,
                    68.5000000000000,
                    64.5000000000000,
                    0,
                    64.5000000000000,
                    67,
                    0,
                    67,
                    67,
                    0,
                    67,
                    67,
                    67,
                    64.5000000000000,
                    68.5000000000000,
                ],
                [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70],
            ]
        ),
        np.array(
            [
                [
                    67,
                    70,
                    67,
                    67,
                    70,
                    68.5000000000000,
                    67,
                    68.5000000000000,
                    64.5000000000000,
                    70,
                    68.5000000000000,
                    67,
                    68.5000000000000,
                    67,
                    70,
                    64.5000000000000,
                ],
                [
                    67,
                    0,
                    67,
                    67,
                    0,
                    64.5000000000000,
                    67,
                    64.5000000000000,
                    68.5000000000000,
                    0,
                    64.5000000000000,
                    67,
                    64.5000000000000,
                    67,
                    0,
                    68.5000000000000,
                ],
                [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70],
            ]
        ),
        np.array(
            [
                [
                    67,
                    70,
                    67,
                    67,
                    70,
                    67,
                    70,
                    68.5000000000000,
                    67,
                    68.5000000000000,
                    64.5000000000000,
                    70,
                    68.5000000000000,
                    67,
                    68.5000000000000,
                    64.5000000000000,
                ],
                [
                    67,
                    0,
                    67,
                    67,
                    0,
                    67,
                    0,
                    64.5000000000000,
                    67,
                    64.5000000000000,
                    68.5000000000000,
                    0,
                    64.5000000000000,
                    67,
                    64.5000000000000,
                    68.5000000000000,
                ],
                [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70],
            ]
        ),
    ),
    axis=2,
)

# matFileNames=['longTrialNoTrigger_forReg_1.mat','longTrialNoTrigger_forReg_2.mat','longTrialNoTrigger_forReg_3.mat','longTrialNoTrigger_forReg_4.mat','longTrialNoTrigger_forReg_5.mat']
# matFieldName='currentAllTrials'

anf_types = ["hsr"]  # select lsr, msr, or hsr for spont rate

# fileSuffix='_ANmodel.pickle'
# fileSuffix = "_ANmodel_correctedLevels.pickle"
fileSuffix = "_ANmodel_maxFs.pickle"

# cfs=(125, 20000, 100)  # This was in the Cochlea repository example
# cfs=(125, 6000, 50)  # 43 comes from the scripting I inherited from Maddox/Vrishab
# cfs = (125, 16000, 43)  # 43 comes from the scripting I inherited from Maddox/Vrishab

fsForModel = 100000  # This is the sampling rate that the Zilany/Carney model needs to run it, so upsample if we don't have that
# fsForRegressor = 5000  # This is the sampling rate we want for the regressor eventually, so downsample at the end of everything
fsForRegressor = 16384  # This is the sampling rate we want for the regressor eventually, so downsample at the end of everything

# numBandsAll = [43, 100]
numBandsAll = [43]


# %%
def doMultANmodels(
    i,
    j,
    k,
    condition,
    regTypes,
    mainDir,
    fsForModel,
    fsForRegressor,
    dBSPLs,
    skipDist,
    anf_types,
    cfs,
    fileSuffix,
):

    ### Make sound
    # fs = 100e3
    # t = np.arange(0, 0.1, 1/fs)
    # s = dsp.chirp(t, 80, t[-1], 20000)
    # s = cochlea.set_dbspl(s, 50)
    # s = np.concatenate( (s, np.zeros(int(10e-3 * fs))) )

    t0 = time.time()

    # stimulus=sp.io.loadmat(f'{regressorDir}{matFileName}')[matFieldName]

    wavFileName = f"{condition}_{regTypes[i]}{k}.wav"

    fs, stimulus = sp.io.wavfile.read(mainDir + wavFileName)
    print(
        f"Wav file has been read in with sampling rate of {fs} resulting in stimulus of shape {stimulus.shape}"
    )
    print("\n")
    # stimulus=stimulus.squeeze()
    # print(stimulus.shape)
    # fs=sp.io.loadmat(f'{regressorDir}{matFileName}')['fs']
    stimulus = sp.signal.resample(stimulus, int(len(stimulus) * fsForModel / fs))

    # Because of copying and pasting from MATLAB, the dimensions don't come in the same order as the loops, so unfortunately it's not [i,j,k]
    stimulus = cochlea.set_dbspl(stimulus, dBSPLs[i, k - 1, j])

    print(
        f"Stimulus has been resampled and normalized resulting in shape {stimulus.shape}"
    )
    print("\n")

    # if regTypes[i]=='distractor' and not np.any(skipDist[j]==k):
    if not np.any(skipDist[j] == k) or regTypes[i] != "distractor":

        ### Run model
        rates = cochlea.run_zilany2014_rate(
            stimulus,
            fsForModel,
            anf_types=anf_types,
            cf=cfs,
            powerlaw="approximate",
            species="human",
        )

        rates = rates.to_numpy()

        rates = rates.mean(
            axis=1
        )  # Uncomment this line to take the average response from all fibers that were modeled

        rates = sp.signal.resample(rates, int(len(rates) * fsForRegressor / fsForModel))

        t1 = time.time()

        print(f"AN model took {t1-t0} seconds to run")
        print(f"AN model to be saved is shape {rates.shape}")
        print("\n")

        eb.save.pickle(rates, f"{mainDir}predictors/{wavFileName[:-4]}{fileSuffix}")

        print(f"Done creating and saving AN model for {wavFileName}")
        print("\n")
        print("\n")


for numBands in numBandsAll:

    # fileSuffix = f"_ANmodel_correctedLevels_{numBands}bands.pickle"

    cfs = (
        125,
        16000,
        numBands,
    )  # 43 comes from the scripting I inherited from Maddox/Vrishab

    for i, mainDir in enumerate(mainDirs):

        for j, condition in enumerate(conditions):

            if doParallel:

                joblib.Parallel(n_jobs=n_jobs, backend=parallelBackend, verbose=49)(
                    joblib.delayed(doMultANmodels)(
                        i,
                        j,
                        k,
                        condition,
                        regTypes,
                        mainDir,
                        fsForModel,
                        fsForRegressor,
                        dBSPLs,
                        skipDist,
                        anf_types,
                        cfs,
                        fileSuffix,
                    )
                    for k in np.arange(1, 17)
                )

            else:

                for k in np.arange(1, 17):
                    doMultANmodels(
                        i,
                        j,
                        k,
                        condition,
                        regTypes,
                        mainDir,
                        fsForModel,
                        fsForRegressor,
                        dBSPLs,
                        skipDist,
                        anf_types,
                        cfs,
                        fileSuffix,
                    )

            ### Plot rates
            # fig, ax = plt.subplots()
            # img = ax.imshow(
            #    rates.T,
            #    aspect='auto'
            # )
            # plt.colorbar(img)
            # plt.show()

# %%
# print(type(rates))
# print(rates.shape)

# %%
# t=np.linspace(0,int(len(rates)/fsForRegressor),len(rates))
# plt.figure()
# plt.plot(t,rates,t,1000*sp.signal.resample(stimulus,len(rates)))

# %%
# test=eb.load.unpickle('/Users/karl/map/stimAndPredictors/targets/predictors/C_target13~gammatone-1.pickle')

# %%
# test.x.shape

# %%
# plt.figure()
# plt.plot(test.x)

# %%

# %%
