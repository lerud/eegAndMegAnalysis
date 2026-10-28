# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy as sp
import mne
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
n_jobs = 16
parallelBackend = "loky"

dBSPL = 75

mainDir = ""

anf_types = ["hsr"]  # select lsr, msr, or hsr for spont rate

fsForModel = 100000  # This is the sampling rate that the Zilany/Carney model needs to run it, so upsample if we don't have that

fsForRegressors = [16384, 5000]

numBandsAll = [43]

outputNames = ["clickANmodelEeg.pickle", "clickANmodelMeg.pickle"]

# filterBeforeModel = None
filterBeforeModel = [20, 4000]
filterPhase = "zero"


t0 = time.time()

wavFileName = f"clickABRstim.wav"

fs, stimulus = sp.io.wavfile.read(mainDir + wavFileName)
print(
    f"Wav file has been read in with sampling rate of {fs} resulting in stimulus of shape {stimulus.shape}"
)

if filterBeforeModel is not None:
    regressorInfo = mne.create_info(ch_names=1, sfreq=fs, ch_types="stim")
    raw = mne.io.RawArray(data=stimulus[np.newaxis, :], info=regressorInfo)
    raw.filter(
        l_freq=filterBeforeModel[0],
        h_freq=filterBeforeModel[1],
        picks="all",
        phase=filterPhase,
    )
    stimulus = raw.get_data().squeeze()

print("\n")
# stimulus=stimulus.squeeze()
# print(stimulus.shape)
# fs=sp.io.loadmat(f'{regressorDir}{matFileName}')['fs']
stimulus = sp.signal.resample(stimulus, int(len(stimulus) * fsForModel / fs))

plt.figure()
plt.plot(stimulus[:100000])

stimulus = cochlea.set_dbspl(stimulus, dBSPL)

print(f"Stimulus has been resampled and normalized resulting in shape {stimulus.shape}")
print("\n")

# cfs = (125, 16000, numBandsAll[0])
cfs = np.linspace(125, 16000, numBandsAll[0])

### Run model


def runOneModelCF(i, cfs, stimulus, fsForModel, anf_types):

    cf = cfs[i]
    rate = cochlea.run_zilany2014_rate(
        stimulus,
        fsForModel,
        anf_types=anf_types,
        cf=cf,
        powerlaw="approximate",
        species="human",
        ffGn=True,
    )
    return rate


rates = joblib.Parallel(n_jobs=n_jobs, backend=parallelBackend, verbose=49)(
    joblib.delayed(runOneModelCF)(
        i,
        cfs,
        stimulus,
        fsForModel,
        anf_types,
    )
    for i in range(len(cfs))
)

# %%
for i in range(len(rates)):
    rates[i] = rates[i].to_numpy()

rates = np.array(rates)

rates = rates.mean(
    axis=0
)  # Uncomment this line to take the average response from all fibers that were modeled

ratesForOutput = []
for i, fsForRegressor in enumerate(fsForRegressors):

    ratesForOutput.append(
        sp.signal.resample(rates, int(len(rates) * fsForRegressor / fsForModel))
    )

    t1 = time.time()

    print(f"AN model took {t1-t0} seconds to run")
    print(f"AN model to be saved is shape {ratesForOutput[i].shape}")
    print("\n")

    plt.figure()
    plt.plot(ratesForOutput[i][0:10000])
    eb.save.pickle(ratesForOutput[i], f"{outputNames[i]}")

    print(
        f"Done creating and saving AN model for {wavFileName} at fs of {fsForRegressor}"
    )
    print("\n")
    print("\n")

# %%
