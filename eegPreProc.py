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
import pathlib
import glob
import mne
import numpy as np
import scipy as sp

# %%
# mainDir = "/Users/karl/map/"
mainDir = "/Volumes/Seagate/map/"

subject = "R3265"

subDirs = "/eegAndMeg/eeg/"
# subDirs='/ffrTests/'

# runName="tones"
# runName="trftrial"
# runName='stacks'
runName = "maintask"
# runName='triggytest-erp'
# runName='triggytest-trf'

refs = ["EXG3", "EXG4"]
freqsToNotch = np.arange(60, 8192, 60)

cropStart = None  # Do this usually
# cropStart = 100  # In units of seconds; for the R3214 dataset for maintask because we started with the Presentation audio settings incorrect. Then we started over, but didn't restart the EEG recording


# %%
def process_initial_raw_eeg(readpath, writepath, refs=None, freqsToNotch=None):
    """
    sets references and lightly filters the raw (.bdf) eeg data. the data are
    then split into two halves and saved to a .mat file to be read into matlab
    for the first denoising stages.

    readpath: path to .bdf file
        (includes filename, e.g., \"~/Desktop/test.bdf\")
    writepath: path to folder to write .mat files
        (does not include filename, e.g., \"~/Desktop/outputs/\". this will save
        two files to that directory: \"~/Desktop/outputs/test_first_half.mat\"
        and \"~/Desktop/outputs/test_second_half.mat\". The directory will be
        created if it doesn't exist.)
    """
    # need to view the channel waveforms to confirm reference channels
    if readpath[-3:] == "bdf":
        raw = mne.io.read_raw_bdf(readpath, preload=True)
    elif readpath[-3:] == "fif":
        raw = mne.io.read_raw_fif(readpath, preload=True)
    else:
        print("Either bdf or fif")
        return

    if refs is not None:
        print(f"Referencing to channels {refs}")
        raw.set_eeg_reference(ref_channels=refs)
        print("\n")

    if freqsToNotch is not None:
        print(f"Applying MNE notch filters at {freqsToNotch} Hz")
        raw.notch_filter(freqsToNotch, "eeg")
        print("\n")

    # get stim channel
    print("getting stim channel")
    s = raw.copy().pick_types(stim=True)
    s = s[:][0]
    print("done")

    # some channels might be bad; drop these
    print("getting raw eeg signal")
    raweeg = raw.copy().pick_types(eeg=True, exclude="bads")
    print("done")

    events = mne.find_events(raw, shortest_event=0)

    print("Printing events array to preview trigger numbers")
    print(events)

    del raw

    # create an object to save to matfile
    print("concatenating data array and stim")
    x = raweeg[:][0]
    x = np.concatenate([x, s], axis=0)
    print(f"preparing to save {x.shape} samples")

    readpath = pathlib.Path(readpath)
    outfname = readpath.stem
    outpath = pathlib.Path(writepath)
    outpath.mkdir(parents=True, exist_ok=True)

    outpath = outpath / outfname

    # was having issues loading the full data set, at least with my version of
    # matlab. for that reason, we split the data into segments. the data are
    # recombined in matlab

    n_segs = 10
    n_samp = (len(raweeg[:][0][0]) // n_segs) * n_segs

    for i in range(10):
        # save each part of data to mat file
        if i <= 8:
            print(f"segment {i} saving {n_samp//n_segs} samples")
            sp.io.savemat(
                f"{str(outpath)}_mat_segment{i}.mat",
                {"data": x[:, i * n_samp // n_segs : (i + 1) * n_samp // n_segs]},
            )
        else:  # i == 9
            # avoid bug where the last few samples (modulo 10) are dropped off since
            # we normally index to (i+1) * n_samp//n_segs which will drop off the
            # last mod n_segs samples
            print(
                f"segment {i} saving last {x[:, i * n_samp//n_segs:].shape[1]} samples"
            )
            sp.io.savemat(
                f"{str(outpath)}_mat_segment{i}.mat",
                {"data": x[:, i * n_samp // n_segs :]},
            )

    return


# %%
try:  # Try to load the raw fif file, which will work if we've made it before
    process_initial_raw_eeg(
        mainDir + subject + subDirs + subject + "_" + runName + ".fif",
        mainDir + subject + subDirs,
        refs=refs,
        freqsToNotch=freqsToNotch,
    )
except:  # if it doesn't exist, we'll make it first, and then load it
    targetFiles = sorted(
        glob.glob(mainDir + subject + subDirs + subject + "_*" + runName + "*.bdf")
    )
    if len(targetFiles) == 1:
        print(f"There is 1 bdf file being processed: {targetFiles[0]}")
        raw = mne.io.read_raw_bdf(targetFiles[0])

        if cropStart is not None:
            raw.crop(tmin=cropStart)

        raw.save(targetFiles[0][:-4] + ".fif")
        process_initial_raw_eeg(
            mainDir + subject + subDirs + subject + "_" + runName + ".fif",
            mainDir + subject + subDirs,
            refs=refs,
            freqsToNotch=freqsToNotch,
        )
    elif len(targetFiles) == 2:
        print(
            f"There are 2 bdf files being concatenated in this order: {targetFiles[0]} and {targetFiles[1]}"
        )
        raw = mne.io.read_raw_bdf(targetFiles[0])
        raw2 = mne.io.read_raw_bdf(targetFiles[1])
        raw.append(raw2)
        raw.save(targetFiles[0][:-4] + ".fif")
        process_initial_raw_eeg(
            mainDir + subject + subDirs + subject + "_" + runName + ".fif",
            mainDir + subject + subDirs,
            refs=refs,
            freqsToNotch=freqsToNotch,
        )
    else:
        print("something is wrong")
