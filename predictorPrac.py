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
"""Generate high-resolution gammatone spectrograms"""
import eelbrain as eb
import trftools
import os
import glob
import numpy as np

# %%
separator = "/"


# stimDir = '/Users/karl/map/stimAndPredictors/targets/'
# stimDir = '/Users/karl/map/stimAndPredictors/distractors/'
stimDir = "/Volumes/Seagate/map/stimAndPredictors/mixes/"

# saveDir = '/Users/karl/map/stimAndPredictors/targets/predictors/'
# saveDir = '/Users/karl/map/stimAndPredictors/distractors/predictors/'
saveDir = "/Volumes/Seagate/map/stimAndPredictors/temp/"

stimNames = sorted(glob.glob(os.path.join(stimDir, "*" + ".wav")))

# %%
for filename in stimNames[:2]:

    stimulusName1 = filename.split(separator)[-1]
    stimulusName = stimulusName1.split(".wav")[0]

    print("Doing " + filename)
    wav = eb.load.wav(filename)
    # gt = trftools.gammatone_bank(wav, 80, 4000, 200, location='left', pad=False)  # This used to say this, taking from trftools, which doesn't work anymore
    gt = eb.gammatone_bank(wav, 80, 4000, 200, location="left")
    gt = eb.resample(gt, 2000)
    eb.save.pickle(gt, os.path.join(saveDir, stimulusName + "~gammatone" + ".pickle"))

# %%

# %%

for filename in stimNames:

    print("Doing " + filename)
    stimulusName1 = filename.split(separator)[-1]
    stimulusName = stimulusName1.split(".wav")[0]

    gt = eb.load.unpickle(os.path.join(saveDir, stimulusName + "~gammatone.pickle"))

    # Remove resampling artifacts
    gt = gt.clip(0, out=gt)

    # apply powerlaw compression
    gt **= 0.6  # maybe do a log compression here instead of power law

    # generate on- and offset detector model
    gt_on = trftools.neural.edge_detector(gt, c=30)

    # 1 band predictors
    eb.save.pickle(
        gt.sum("frequency"),
        os.path.join(saveDir, stimulusName + "~gammatone-1" + ".pickle"),
    )
    eb.save.pickle(
        gt_on.sum("frequency"),
        os.path.join(saveDir, stimulusName + "~gammatone-on-1" + ".pickle"),
    )

    # 8 band predictors
    x = gt.bin(nbins=8, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-8.pickle"))

    x = gt_on.bin(nbins=8, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-on-8.pickle"))

    # 24 band predictors
    x = gt.bin(nbins=25, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-25.pickle"))

    x = gt_on.bin(nbins=25, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-on-25.pickle"))

    # 24 band predictors
    x = gt.bin(nbins=50, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-50.pickle"))

    x = gt_on.bin(nbins=50, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-on-50.pickle"))

    # 24 band predictors
    x = gt.bin(nbins=100, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-100.pickle"))

    x = gt_on.bin(nbins=100, func=np.sum, dim="frequency")
    eb.save.pickle(x, os.path.join(saveDir, stimulusName + "~gammatone-on-100.pickle"))


# %%
# ons2k=eb.load.unpickle('audioMixes/pilot_LEGACY/fs2000/A_target7|gammatone-on-1.pickle')
# ons1k=eb.load.unpickle('audioMixes/pilot_LEGACY/A_target7|gammatone-on-1.pickle')

# env=eb.load.unpickle('audioMixes/pilot_LEGACY/fs2000/A_target7_gammatone-1.pickle')
# wav=eb.load.wav('audioMixes/pilot_LEGACY/A_target7.wav')


# %%
# eb.plot.UTS([wav,env,ons],xlim=[3,5])
# eb.plot.UTS(env,xlim=5)
# eb.plot.UTS([ons2k,ons1k],xlim=[3,5])

# %%
