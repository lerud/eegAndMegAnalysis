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
import os
import glob
import mne
import numpy as np

# %%
doEeg = True
doMeg = True

# cropStart = None  # Do this usually
# cropStart = 100  # In units of seconds; for the R3214 dataset for maintask because we started with the Presentation audio settings incorrect. Then we started over, but didn't restart the EEG recording
# These crops are for taking an amount of time off of the BEGINNING of an acquisition. I changed this after R3214 so it may not work exactly right for that subject
cropStartsEeg = {"R3214": {"maintask": 100}, "R3362": {"maintask": 100}}
cropStartsMeg = {"R3362": {"maintask": 100}}

mainDir = "/Volumes/Seagate/map/"

subjects = ["R3362"]
# subjects = ["R3313", "R3317", "R3320"]

subDirsEeg = "/eegAndMeg/eeg/"
subDirsMeg = "/eegAndMeg/meg/"
# subDirs='/ffrTests/'

# runName="tones"
runNamesEeg = ["trftrial", "tones", "maintask", "baseline"]
runNamesMeg = ["trftrial", "tones", "maintask", "baseline", "emptyroom"]
# runName='stacks'
# runName = "maintask"
# runName='triggytest-erp'
# runName='triggytest-trf'

visit = ""

# %%
loc = np.array(
    [
        0.09603,
        -0.07437,
        0.00905,
        -0.5447052,
        -0.83848277,
        0.01558496,
        0.0,
        -0.01858388,
        -0.9998273,
        0.8386276,
        -0.54461113,
        0.01012274,
    ]
)


def changeSensorAndSave(raw, mainDir, subDirsMeg, task, visit, subj, loc):
    fileDir = mainDir + subj + subDirsMeg + subj[-5:]
    file = fileDir + "_" + task + visit + "-raw.fif"
    path = os.path.join(fileDir, file)
    # raw = mne.io.read_raw_fif(path, preload=True)
    index = raw.ch_names.index("MEG 056")
    raw.info["chs"][index]["loc"] = loc
    raw.save(path, overwrite=True)
    print("Done with", task, subj)


# %%
if doEeg:
    for subject in subjects:
        for runName in runNamesEeg:
            targetFiles = sorted(
                glob.glob(
                    mainDir
                    + subject
                    + subDirsEeg
                    + subject[-5:]
                    + "_"
                    + runName
                    + "*.bdf"
                )
            )
            if len(targetFiles) == 1:
                print(f"There is 1 bdf file being processed: {targetFiles[0]}")
                raw = mne.io.read_raw_bdf(targetFiles[0])

                if subject in cropStartsEeg:
                    if runName in cropStartsEeg[subject]:
                        cropStart = cropStartsEeg[subject][runName]
                        raw.crop(tmin=cropStart)

                # A block to find and print events to get an idea whether it looks correct
                events = mne.find_events(raw, shortest_event=0)
                print("\n")
                print("Here is a sample of the initial events matrix:")
                print("\n")
                print(events[:30, :])
                print(events[-30:, :])
                events[:, 2] = events[:, 2] - events[:, 1]
                print("\n")
                print("\n")
                print(
                    "Here is a sample of the events matrix after subtracting the middle column:"
                )
                print("\n")
                print(events[:30, :])
                print(events[-30:, :])

                raw.save(targetFiles[0][:-4] + ".fif")
            elif len(targetFiles) == 2:
                print(
                    f"There are 2 bdf files being concatenated in this order: {targetFiles[0]} and {targetFiles[1]}"
                )
                raw = mne.io.read_raw_bdf(targetFiles[0])

                if subject in cropStartsEeg:
                    if runName in cropStartsEeg[subject]:
                        cropStart = cropStartsEeg[subject][runName]
                        raw.crop(tmin=cropStart)

                raw2 = mne.io.read_raw_bdf(targetFiles[1])
                raw.append(raw2)

                # A block to find and print events to get an idea whether it looks correct
                events = mne.find_events(raw, shortest_event=0)
                print("\n")
                print("Here is a sample of the initial events matrix:")
                print("\n")
                print(events[:30, :])
                print(events[-30:, :])
                events[:, 2] = events[:, 2] - events[:, 1]
                print("\n")
                print("\n")
                print(
                    "Here is a sample of the events matrix after subtracting the middle column:"
                )
                print("\n")
                print(events[:30, :])
                print(events[-30:, :])

                raw.save(targetFiles[0][:-4] + ".fif")
            else:
                print("error, something is wrong")

# %%
if doMeg:
    for subject in subjects:
        for runName in runNamesMeg:
            targetFiles = sorted(
                glob.glob(
                    mainDir
                    + subject
                    + subDirsMeg
                    + subject[-5:]
                    + "_"
                    + runName
                    + "*-raw.fif"
                )  # This will capture e.g. blah_maintask-raw.fif and blah_maintask2-raw.fif but NOT e.g. blah_maintask-raw-1.fif, which are just subparts of the main fif file
            )
            if len(targetFiles) == 1:
                print(f"There is 1 fif file being processed: {targetFiles[0]}")
                raw = mne.io.read_raw_fif(targetFiles[0], preload=True)

                if subject in cropStartsMeg:
                    if runName in cropStartsMeg[subject]:
                        cropStart = cropStartsMeg[subject][runName]
                        raw.crop(tmin=cropStart)

            elif len(targetFiles) > 1:
                print(
                    f"There are {len(targetFiles)} fif files being concatenated in this order: {targetFiles}"
                )
                raw = mne.io.read_raw_fif(targetFiles[0])

                if subject in cropStartsMeg:
                    if runName in cropStartsMeg[subject]:
                        cropStart = cropStartsMeg[subject][runName]
                        raw.crop(tmin=cropStart)

                for i in range(
                    len(targetFiles) - 1
                ):  # Loop over the remaining number of fif files after the first one
                    rawNext = mne.io.read_raw_fif(
                        targetFiles[i + 1], preload=True
                    )  # and use the index plus one because it starts at zero
                    raw.append(rawNext)
            else:
                print("error, something is wrong")

            changeSensorAndSave(raw, mainDir, subDirsMeg, runName, visit, subject, loc)
