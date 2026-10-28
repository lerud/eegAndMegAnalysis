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
import mne
import numpy as np
import eelbrain as eb
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
from nilearn import plotting

matplotlib.use("QtAgg")
plt.ion()

# %%
useAvgBrain = False

# subject = "R3045";badChanList=None;condition="A";presStopCorrection=0
# subject='R3089';badChanList=['P7','CP6','C4','T7','CP5','P3','P4','O2','Oz','PO4'];condition='B';presStopCorrection=0
# subject='R3093';badChanList=['Oz','P8','CP6','Fp2'];condition='C';presStopCorrection=0
# subject='R3095';badChanList=['P7','T8','O2','PO4'];condition='D';presStopCorrection=0  # and possibly O2 and PO4

# subject='R3151';badChanList=['C3','FC5','P4'];condition='A';useAvgBrain=True
# subject='R2774';badChanList=['Fp1','AF3','F7','F3','Fz','F4','FC6','C3','CP5','Pz','CP6','P8','FC1','FC5','T7','AF4'];condition='B'
# subject='R3152';badChanList=['T7','C3','P7','Pz','O1','P8','CP6'];condition='C'
subject = "R2877"
# badChanList = ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"]
badChanList = None
condition = "D"
useAvgBrain = True

# subject='R3157';badChanList=['CP6'];condition='A'  # Keep an eye on CP2 and possibly others
# subject='R3157';badChanList=None;condition='A'  # Keep an eye on CP2 and possibly others
# subject='R2783';badChanList=['T7','P7','CP5'];condition='B'
# subject='R2783';badChanList=None;condition='B'


if useAvgBrain:
    mriSubject = "fsaverage"
else:
    mriSubject = subject

subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

subDirs = "/eegAndMeg/eeg/"

eegLocation = "/Users/karl/map/" + subject + subDirs


nameOfRegressor = "_ANmodel_correctedLevels"
initial_time = 0.0085
# nameOfRegressor='~gammatone-1';initial_time = 0.05
# nameOfRegressor='~gammatone-on-1';initial_time = 0.05

typeOfRegressor = "mix"
# typeOfRegressor = "target"
# typeOfRegressor='distractor'

baselineFiles = sorted(glob.glob(f"{eegLocation}*baseline*bdf"))
bdfFile = baselineFiles[0]

# These are bandpasses for the noise covariance matrix, and they matter! Probably should be the same as the bandpasses on the TRFs
# l_freq = 2
# h_freq = None
l_freq = 20
h_freq = 1000

refs = ["EXG3", "EXG4"]
notchFreqs = np.arange(60, 8192, 60)

labels_vol = ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Brain-Stem"]

evoked = mne.read_evokeds(
    f"{eegLocation}evoked{nameOfRegressor}_{typeOfRegressor}-ave.fif"
)
evoked = evoked[0]
evoked.set_eeg_reference(projection=True)


# %%
baselineRaw = mne.io.read_raw_bdf(bdfFile, preload=True)

baselineRaw.set_eeg_reference(ref_channels=refs)
baselineRaw.notch_filter(notchFreqs)

# %%
elp_ch_names = [
    "Mark1",
    "Mark2",
    "Mark3",
    "Mark4",
    "Mark5",
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
]

channelFile = glob.glob("/Users/karl/map/" + subject + "/digitization/*eeg*.elp")
if len(channelFile) == 0:
    channelFile = glob.glob("/Users/karl/map/" + subject + "/digitization/*EEG*.elp")
print(f"Using the channel file {channelFile[0]}")

digMontage = mne.channels.read_dig_polhemus_isotrak(
    channelFile[0], ch_names=elp_ch_names
)
changeChTypes = {
    "EXG1": "eog",
    "EXG2": "eog",
    "EXG3": "eog",
    "EXG4": "eog",
    "EXG5": "eog",
    "EXG6": "eog",
    "EXG7": "eog",
    "EXG8": "eog",
}
elpChangeChTypes = {
    "Mark1": "hpi",
    "Mark2": "hpi",
    "Mark3": "hpi",
    "Mark4": "hpi",
    "Mark5": "hpi",
}

baselineRaw.set_channel_types(changeChTypes)
baselineRaw.set_montage(digMontage)

if badChanList is not None:
    baselineRaw.info["bads"].extend(badChanList)
    # baselineRaw.interpolate_bads()

# %%
if l_freq is not None or h_freq is not None:
    # denoised.filter(l_freq=.1,h_freq=20)
    # denoised.filter(l_freq=10,h_freq=50)
    baselineRaw.filter(l_freq=l_freq, h_freq=h_freq)
    print("\n")

# %%
baselineCov = mne.compute_raw_covariance(baselineRaw, picks="eeg")

# %%
if os.path.exists(subjects_dir + "/" + mriSubject + "/bem/conductorModel.pickle"):
    bem = eb.load.unpickle(
        subjects_dir + "/" + mriSubject + "/bem/conductorModel.pickle"
    )
else:
    surfaces = mne.make_bem_model(mriSubject)
    bem = mne.make_bem_solution(surfaces)
    eb.save.pickle(bem, subjects_dir + "/" + mriSubject + "/bem/conductorModel.pickle")

trans = eegLocation + "eeg-trans.fif"
mriAseg = subjects_dir + "/" + mriSubject + "/mri/aseg.mgz"

src_surf = mne.setup_source_space(
    mriSubject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir
)

src_vol = mne.setup_volume_source_space(
    mriSubject,
    mri=mriAseg,
    pos=10.0,
    bem=bem,
    volume_label=labels_vol,
    subjects_dir=subjects_dir,
    add_interpolator=True,  # just for speed, usually this should be True
    verbose=True,
)

# Generate the mixed source space
src_mix = src_surf + src_vol
print(
    f"The mixed source space contains {len(src_mix)} spaces and "
    f"{sum(s['nuse'] for s in src_mix)} vertices"
)

# src.plot(subjects_dir=subjects_dir)

# Setup volumn source space (This is the part how to create volSourceEstimate)
# vol_src = mne.setup_volume_source_space(
#     subject, mri=mri, pos=10.0, bem=bem,
#     subjects_dir=subjects_dir,
#     add_interpolator=True,
#     verbose=True)

# %%
fwd_surf = mne.make_forward_solution(
    evoked.info,
    trans,
    src_surf,
    bem,
    mindist=5.0,  # ignore sources<=5mm from innerskull
    meg=False,
    eeg=True,
    n_jobs=None,
)
fwd_vol = mne.make_forward_solution(
    evoked.info,
    trans,
    src_vol,
    bem,
    mindist=5.0,  # ignore sources<=5mm from innerskull
    meg=False,
    eeg=True,
    n_jobs=None,
)
fwd_mix = mne.make_forward_solution(
    evoked.info,
    trans,
    src_mix,
    bem,
    mindist=5.0,  # ignore sources<=5mm from innerskull
    meg=False,
    eeg=True,
    n_jobs=None,
)

# del src  # save memory

leadfield = fwd_mix["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
print(
    f"The fwd source space contains {len(fwd_mix['src'])} spaces and "
    f"{sum(s['nuse'] for s in fwd_mix['src'])} vertices"
)


# %%
snr = 3.0  # use smaller SNR for raw data
inv_method = "dSPM"  # sLORETA, MNE, dSPM
parc = "aparc"  # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
loose_surf = dict(surface=0.2)
loose_vol = dict(volume=1.0)
loose_mix = dict(surface=0.2, volume=1.0)
depth = 0.8
# depth = 10.0

lambda2 = 1.0 / snr**2

inverse_operator_surf = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd_surf, baselineCov, depth=depth, loose=loose_surf, verbose=True
)

inverse_operator_vol = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd_vol, baselineCov, depth=depth, loose=loose_vol, verbose=True
)

inverse_operator_mix = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd_mix, baselineCov, depth=depth, loose=loose_mix, verbose=True
)

# del fwd

stc = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator_surf, lambda2, inv_method, pick_ori="normal"
)
# stc = mne.minimum_norm.apply_inverse(
#     evoked, inverse_operator, lambda2, inv_method, pick_ori=None
# )

# src = inverse_operator["src"]

# %%
stc_vec = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator_mix, lambda2, inv_method, pick_ori="vector"
)
brain = stc_vec.plot(
    hemi="both",
    src=inverse_operator_mix["src"],
    views="coronal",
    initial_time=initial_time,
    subjects_dir=subjects_dir,
    brain_kwargs=dict(silhouette=True),
    smoothing_steps=7,
    show_traces=True,
)

# %%
# brain2 = stc.surface().plot(
#     initial_time=initial_time, subjects_dir=subjects_dir, smoothing_steps=7
# )

# %%
# fig = stc.volume().plot(initial_time=initial_time, src=src, subjects_dir=subjects_dir)

# %%

# Plot electrode locations on scalp
fig = mne.viz.plot_alignment(
    evoked.info,
    trans,
    subject=mriSubject,
    dig=True,
    eeg=["original", "projected"],
    meg=[],
    coord_frame="head",
    subjects_dir=subjects_dir,
    surfaces=dict(brain=0.4, outer_skull=0.6, inner_skull=0.4, head=None),
)

# Set viewing angle
mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)


# %%
def calcPC(mat, doPC):
    if doPC:
        outmat = np.zeros((mat.shape[2], mat.shape[0]))
        for i in range(mat.shape[0]):
            submat = mat[i, :, :].T
            u, s, _ = np.linalg.svd(submat)
            PC = u[:, 0] * s[0]
            pearsonR = np.corrcoef(PC, submat.mean(axis=1))[1, 0]
            print(pearsonR)
            PC *= np.sign(pearsonR)
            outmat[:, i] = PC
    else:
        outmat = mat.mean(axis=1).T
    return outmat


# %%
doPC = True

nonVecMode = "mean_flip"
doVec = True


if doVec:

    allSourceTimeCourses = []

    audLabels = mne.read_labels_from_annot(mriSubject, regexp="transversetemporal")
    audTimeCourses = stc_vec.extract_label_time_course(audLabels, src_mix)[0:2, :, :]
    print(audTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(audTimeCourses, doPC))

    tempLabels = mne.read_labels_from_annot(mriSubject, regexp="superiortemporal")
    tempTimeCourses = stc_vec.extract_label_time_course(tempLabels, src_mix)[0:2, :, :]
    print(tempTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(tempTimeCourses, doPC))

    supParLabels = mne.read_labels_from_annot(mriSubject, regexp="superiorparietal")
    supParTimeCourses = stc_vec.extract_label_time_course(supParLabels, src_mix)[
        0:2, :, :
    ]
    print(supParTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(supParTimeCourses, doPC))

    infParLabels = mne.read_labels_from_annot(mriSubject, regexp="inferiorparietal")
    infParTimeCourses = stc_vec.extract_label_time_course(infParLabels, src_mix)[
        0:2, :, :
    ]
    print(infParTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(infParTimeCourses, doPC))

    frontLabels = mne.read_labels_from_annot(mriSubject, regexp="frontalpole")
    frontTimeCourses = stc_vec.extract_label_time_course(frontLabels, src_mix)[
        0:2, :, :
    ]
    print(frontTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(frontTimeCourses, doPC))

    postcLabels = mne.read_labels_from_annot(mriSubject, regexp="postcentral")
    postcTimeCourses = stc_vec.extract_label_time_course(postcLabels, src_mix)[
        0:2, :, :
    ]
    print(postcTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(postcTimeCourses, doPC))

    occLabels = mne.read_labels_from_annot(mriSubject, regexp="lateraloccipital")
    occTimeCourses = stc_vec.extract_label_time_course(occLabels, src_mix)[0:2, :, :]
    print(occTimeCourses.shape)
    allSourceTimeCourses.append(calcPC(occTimeCourses, doPC))

    subcortLabels = []
    subcortTimeCourses = stc_vec.extract_label_time_course(subcortLabels, src_mix)
    print(subcortTimeCourses.shape)

    ymax = np.max(
        [calcPC(subcortTimeCourses, doPC).max(), np.array(allSourceTimeCourses).max()]
    )
    ymin = np.min(
        [calcPC(subcortTimeCourses, doPC).min(), np.array(allSourceTimeCourses).min()]
    )

    figsize = [20, 11]
    fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
        nrows=2, ncols=4, figsize=figsize
    )

    ax1.plot(stc._times, calcPC(audTimeCourses, doPC))
    # ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
    ax1.set_ylim([ymin, ymax])
    ax1.legend([audLabels[0].name, audLabels[1].name])

    ax2.plot(stc._times, calcPC(tempTimeCourses, doPC))
    ax2.set_ylim([ymin, ymax])
    ax2.legend([tempLabels[0].name, tempLabels[1].name])

    ax3.plot(stc._times, calcPC(supParTimeCourses, doPC))
    ax3.set_ylim([ymin, ymax])
    ax3.legend([supParLabels[0].name, supParLabels[1].name])

    ax4.plot(stc._times, calcPC(infParTimeCourses, doPC))
    ax4.set_ylim([ymin, ymax])
    ax4.legend([infParLabels[0].name, infParLabels[1].name])

    ax5.plot(stc._times, calcPC(frontTimeCourses, doPC))
    ax5.set_ylim([ymin, ymax])
    ax5.legend([frontLabels[0].name, frontLabels[1].name])

    ax6.plot(stc._times, calcPC(postcTimeCourses, doPC))
    ax6.set_ylim([ymin, ymax])
    ax6.legend([postcLabels[0].name, postcLabels[1].name])

    ax7.plot(stc._times, calcPC(occTimeCourses, doPC))
    ax7.set_ylim([ymin, ymax])
    ax7.legend([occLabels[0].name, occLabels[1].name])

    ax8.plot(stc._times, calcPC(subcortTimeCourses, doPC))
    # ax8.plot(stc._times,subcortTimeCourses[:,1,:].T)
    # ax8.set_ylim([ymin, ymax])
    ax8.legend(labels_vol)

else:

    allSourceTimeCourses = []

    audLabels = mne.read_labels_from_annot(mriSubject, regexp="transversetemporal")
    audTimeCourses = stc.extract_label_time_course(audLabels, src_surf, mode=nonVecMode)
    print(audTimeCourses.shape)
    allSourceTimeCourses.append(audTimeCourses[0:2, :].T)

    tempLabels = mne.read_labels_from_annot(mriSubject, regexp="superiortemporal")
    tempTimeCourses = stc.extract_label_time_course(
        tempLabels, src_surf, mode=nonVecMode
    )
    print(tempTimeCourses.shape)
    allSourceTimeCourses.append(tempTimeCourses[0:2, :].T)

    supParLabels = mne.read_labels_from_annot(mriSubject, regexp="superiorparietal")
    supParTimeCourses = stc.extract_label_time_course(
        supParLabels, src_surf, mode=nonVecMode
    )
    print(supParTimeCourses.shape)
    allSourceTimeCourses.append(supParTimeCourses[0:2, :].T)

    infParLabels = mne.read_labels_from_annot(mriSubject, regexp="inferiorparietal")
    infParTimeCourses = stc.extract_label_time_course(
        infParLabels, src_surf, mode=nonVecMode
    )
    print(infParTimeCourses.shape)
    allSourceTimeCourses.append(infParTimeCourses[0:2, :].T)

    frontLabels = mne.read_labels_from_annot(mriSubject, regexp="frontalpole")
    frontTimeCourses = stc.extract_label_time_course(
        frontLabels, src_surf, mode=nonVecMode
    )
    print(frontTimeCourses.shape)
    allSourceTimeCourses.append(frontTimeCourses[0:2, :].T)

    postcLabels = mne.read_labels_from_annot(mriSubject, regexp="postcentral")
    postcTimeCourses = stc.extract_label_time_course(
        postcLabels, src_surf, mode=nonVecMode
    )
    print(postcTimeCourses.shape)
    allSourceTimeCourses.append(postcTimeCourses[0:2, :].T)

    occLabels = mne.read_labels_from_annot(mriSubject, regexp="lateraloccipital")
    occTimeCourses = stc.extract_label_time_course(occLabels, src_surf, mode=nonVecMode)
    print(occTimeCourses.shape)
    allSourceTimeCourses.append(occTimeCourses[0:2, :].T)

    subcortLabels = []
    subcortTimeCourses = stc_vec.extract_label_time_course(subcortLabels, src_mix)
    print(subcortTimeCourses.shape)

    ymax = np.max(
        [subcortTimeCourses.mean(axis=1).T.max(), np.array(allSourceTimeCourses).max()]
    )
    ymin = np.min(
        [subcortTimeCourses.mean(axis=1).T.min(), np.array(allSourceTimeCourses).min()]
    )

    figsize = [20, 11]
    fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
        nrows=2, ncols=4, figsize=figsize
    )

    ax1.plot(stc._times, audTimeCourses[0:2, :].T)
    ax1.set_ylim([ymin, ymax])
    ax1.legend([audLabels[0].name, audLabels[1].name])

    ax2.plot(stc._times, tempTimeCourses[0:2, :].T)
    ax2.set_ylim([ymin, ymax])
    ax2.legend([tempLabels[0].name, tempLabels[1].name])

    ax3.plot(stc._times, supParTimeCourses[0:2, :].T)
    ax3.set_ylim([ymin, ymax])
    ax3.legend([supParLabels[0].name, supParLabels[1].name])

    ax4.plot(stc._times, infParTimeCourses[0:2, :].T)
    ax4.set_ylim([ymin, ymax])
    ax4.legend([infParLabels[0].name, infParLabels[1].name])

    ax5.plot(stc._times, frontTimeCourses[0:2, :].T)
    ax5.set_ylim([ymin, ymax])
    ax5.legend([frontLabels[0].name, frontLabels[1].name])

    ax6.plot(stc._times, postcTimeCourses[0:2, :].T)
    ax6.set_ylim([ymin, ymax])
    ax6.legend([postcLabels[0].name, postcLabels[1].name])

    ax7.plot(stc._times, occTimeCourses[0:2, :].T)
    ax7.set_ylim([ymin, ymax])
    ax7.legend([occLabels[0].name, occLabels[1].name])

    ax8.plot(stc._times, subcortTimeCourses.mean(axis=1).T)
    # ax8.plot(stc._times,subcortTimeCourses[:,2,:].T)
    # ax8.set_ylim([ymin, ymax])
    ax8.legend(labels_vol)


# %%
