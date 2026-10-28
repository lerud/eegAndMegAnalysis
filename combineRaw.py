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

mainDir = "/Users/karl/map/"
subject = "R3095"
rawDir = "/eegAndMeg/meg/"
rawName1 = subject + "_maintask1-raw.fif"
rawName2 = subject + "_maintask2-raw.fif"
outName = subject + "_maintask_complete-raw.fif"

raw = mne.io.read_raw_fif(mainDir + subject + rawDir + rawName1)
raw2 = mne.io.read_raw_fif(mainDir + subject + rawDir + rawName2)
raw.append(raw2)
raw.save(mainDir + subject + rawDir + outName, overwrite=True)
