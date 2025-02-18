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
import sys
from nilearn import plotting

matplotlib.use("QtAgg")
plt.ion()


# %%
# brain = stc_vec.plot(
#     hemi="both",
#     src=inverse_operator_mix["src"],
#     views="coronal",
#     initial_time=initial_time,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=7,
#     show_traces=True
# )

# %%
# brain2 = stc.surface().plot(
#     initial_time=initial_time, subjects_dir=subjects_dir, smoothing_steps=7
# )

# %%
# fig = stc.volume().plot(initial_time=initial_time, src=src, subjects_dir=subjects_dir)

# %%

# # Plot electrode locations on scalp
# fig = mne.viz.plot_alignment(
#     evoked.info,
#     trans,
#     subject=mriSubject,
#     dig=True,
#     eeg=["original", "projected"],
#     meg=[],
#     coord_frame="head",
#     subjects_dir=subjects_dir,
#     surfaces=dict(brain=0.4, outer_skull=0.6,inner_skull=0.4, head=None)
# )

# # Set viewing angle
# mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)


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

# %%


def extractSourcesSingleCondition(
    subjectsToAverage,
    subDirs,
    nameOfRegressor,
    typeOfRegressor,
    filenameSuffix,
    doPC,
    nonVecMode,
    doVec,
):

    audTimeCoursesAllSubjs = []
    tempTimeCoursesAllSubjs = []
    supParTimeCoursesAllSubjs = []
    infParTimeCoursesAllSubjs = []
    frontTimeCoursesAllSubjs = []
    postcTimeCoursesAllSubjs = []
    occTimeCoursesAllSubjs = []
    subcortTimeCoursesAllSubjs = []
    allSourceTimeCoursesAllSubjs = []

    for i, subject in enumerate(subjectsToAverage):

        # eegLocation = "/Users/karl/map/" + subject + subDirs
        eegLocation = "/Volumes/Seagate/map/" + subject + subDirs

        # (
        #     subject,
        #     mriSubject,
        #     labels_vol,
        #     src_surf,
        #     src_vol,
        #     src_mix,
        #     fwd_surf,
        #     fwd_vol,
        #     fwd_mix,
        #     inverse_operator_surf,
        #     inverse_operator_vol,
        #     inverse_operator_mix,
        #     stc,
        #     stc_vec,
        # ) = eb.load.unpickle(
        #     f"{eegLocation}sources{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle"
        # )

        (
            subject,
            mriSubject,
            labels_vol,
            src_surf,
            src_vol,
            src_mix,
            fwd_surf,
            fwd_vol,
            fwd_mix,
            stc_vec,
        ) = eb.load.unpickle(
            f"{eegLocation}sources{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle"
        )

        if doVec:

            allSourceTimeCourses = []

            audLabels = mne.read_labels_from_annot(
                mriSubject, regexp="transversetemporal"
            )
            audTimeCourses = calcPC(
                stc_vec.extract_label_time_course(audLabels, src_mix)[0:2, :, :], doPC
            )
            print(audTimeCourses.shape)
            allSourceTimeCourses.append(audTimeCourses)

            tempLabels = mne.read_labels_from_annot(
                mriSubject, regexp="superiortemporal"
            )
            tempTimeCourses = calcPC(
                stc_vec.extract_label_time_course(tempLabels, src_mix)[0:2, :, :], doPC
            )
            print(tempTimeCourses.shape)
            allSourceTimeCourses.append(tempTimeCourses)

            supParLabels = mne.read_labels_from_annot(
                mriSubject, regexp="superiorparietal"
            )
            supParTimeCourses = calcPC(
                stc_vec.extract_label_time_course(supParLabels, src_mix)[0:2, :, :],
                doPC,
            )
            print(supParTimeCourses.shape)
            allSourceTimeCourses.append(supParTimeCourses)

            infParLabels = mne.read_labels_from_annot(
                mriSubject, regexp="inferiorparietal"
            )
            infParTimeCourses = calcPC(
                stc_vec.extract_label_time_course(infParLabels, src_mix)[0:2, :, :],
                doPC,
            )
            print(infParTimeCourses.shape)
            allSourceTimeCourses.append(infParTimeCourses)

            frontLabels = mne.read_labels_from_annot(mriSubject, regexp="frontalpole")
            frontTimeCourses = calcPC(
                stc_vec.extract_label_time_course(frontLabels, src_mix)[0:2, :, :], doPC
            )
            print(frontTimeCourses.shape)
            allSourceTimeCourses.append(frontTimeCourses)

            postcLabels = mne.read_labels_from_annot(mriSubject, regexp="postcentral")
            postcTimeCourses = calcPC(
                stc_vec.extract_label_time_course(postcLabels, src_mix)[0:2, :, :], doPC
            )
            print(postcTimeCourses.shape)
            allSourceTimeCourses.append(postcTimeCourses)

            occLabels = mne.read_labels_from_annot(
                mriSubject, regexp="lateraloccipital"
            )
            occTimeCourses = calcPC(
                stc_vec.extract_label_time_course(occLabels, src_mix)[0:2, :, :], doPC
            )
            print(occTimeCourses.shape)
            allSourceTimeCourses.append(occTimeCourses)

            subcortLabels = []
            subcortTimeCourses = calcPC(
                stc_vec.extract_label_time_course(subcortLabels, src_mix), doPC
            )
            print(subcortTimeCourses.shape)

            allSourceTimeCourses = np.array(allSourceTimeCourses)

            audTimeCoursesAllSubjs.append(audTimeCourses)
            tempTimeCoursesAllSubjs.append(tempTimeCourses)
            supParTimeCoursesAllSubjs.append(supParTimeCourses)
            infParTimeCoursesAllSubjs.append(infParTimeCourses)
            frontTimeCoursesAllSubjs.append(frontTimeCourses)
            postcTimeCoursesAllSubjs.append(postcTimeCourses)
            occTimeCoursesAllSubjs.append(occTimeCourses)
            subcortTimeCoursesAllSubjs.append(subcortTimeCourses)
            allSourceTimeCoursesAllSubjs.append(allSourceTimeCourses)

        else:

            allSourceTimeCourses = []

            audLabels = mne.read_labels_from_annot(
                mriSubject, regexp="transversetemporal"
            )
            audTimeCourses = stc.extract_label_time_course(
                audLabels, src_surf, mode=nonVecMode
            ).T
            print(audTimeCourses.shape)
            allSourceTimeCourses.append(audTimeCourses)

            tempLabels = mne.read_labels_from_annot(
                mriSubject, regexp="superiortemporal"
            )
            tempTimeCourses = stc.extract_label_time_course(
                tempLabels, src_surf, mode=nonVecMode
            ).T
            print(tempTimeCourses.shape)
            allSourceTimeCourses.append(tempTimeCourses)

            supParLabels = mne.read_labels_from_annot(
                mriSubject, regexp="superiorparietal"
            )
            supParTimeCourses = stc.extract_label_time_course(
                supParLabels, src_surf, mode=nonVecMode
            ).T
            print(supParTimeCourses.shape)
            allSourceTimeCourses.append(supParTimeCourses)

            infParLabels = mne.read_labels_from_annot(
                mriSubject, regexp="inferiorparietal"
            )
            infParTimeCourses = stc.extract_label_time_course(
                infParLabels, src_surf, mode=nonVecMode
            ).T
            print(infParTimeCourses.shape)
            allSourceTimeCourses.append(infParTimeCourses)

            frontLabels = mne.read_labels_from_annot(mriSubject, regexp="frontalpole")
            frontTimeCourses = stc.extract_label_time_course(
                frontLabels, src_surf, mode=nonVecMode
            ).T
            print(frontTimeCourses.shape)
            allSourceTimeCourses.append(frontTimeCourses)

            postcLabels = mne.read_labels_from_annot(mriSubject, regexp="postcentral")
            postcTimeCourses = stc.extract_label_time_course(
                postcLabels, src_surf, mode=nonVecMode
            ).T
            print(postcTimeCourses.shape)
            allSourceTimeCourses.append(postcTimeCourses)

            occLabels = mne.read_labels_from_annot(
                mriSubject, regexp="lateraloccipital"
            )
            occTimeCourses = stc.extract_label_time_course(
                occLabels, src_surf, mode=nonVecMode
            ).T
            print(occTimeCourses.shape)
            allSourceTimeCourses.append(occTimeCourses)

            subcortLabels = []
            subcortTimeCourses = calcPC(
                stc_vec.extract_label_time_course(subcortLabels, src_mix), doPC
            )
            print(subcortTimeCourses.shape)

            allSourceTimeCourses = np.array(allSourceTimeCourses)

            audTimeCoursesAllSubjs.append(audTimeCourses)
            tempTimeCoursesAllSubjs.append(tempTimeCourses)
            supParTimeCoursesAllSubjs.append(supParTimeCourses)
            infParTimeCoursesAllSubjs.append(infParTimeCourses)
            frontTimeCoursesAllSubjs.append(frontTimeCourses)
            postcTimeCoursesAllSubjs.append(postcTimeCourses)
            occTimeCoursesAllSubjs.append(occTimeCourses)
            subcortTimeCoursesAllSubjs.append(subcortTimeCourses)
            allSourceTimeCoursesAllSubjs.append(allSourceTimeCourses)

    #         ymax = np.max([subcortTimeCourses.mean(axis=1).T.max(), np.array(allSourceTimeCourses).max()])
    #         ymin = np.min([subcortTimeCourses.mean(axis=1).T.min(), np.array(allSourceTimeCourses).min()])

    #         figsize = [20,11]
    #         fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
    #                 nrows=2, ncols=4, figsize=figsize)

    #         ax1.plot(stc._times,audTimeCourses[0:2,:].T)
    #         ax1.set_ylim([ymin, ymax])
    #         ax1.legend([audLabels[0].name, audLabels[1].name])

    #         ax2.plot(stc._times,tempTimeCourses[0:2,:].T)
    #         ax2.set_ylim([ymin, ymax])
    #         ax2.legend([tempLabels[0].name, tempLabels[1].name])

    #         ax3.plot(stc._times,supParTimeCourses[0:2,:].T)
    #         ax3.set_ylim([ymin, ymax])
    #         ax3.legend([supParLabels[0].name, supParLabels[1].name])

    #         ax4.plot(stc._times,infParTimeCourses[0:2,:].T)
    #         ax4.set_ylim([ymin, ymax])
    #         ax4.legend([infParLabels[0].name, infParLabels[1].name])

    #         ax5.plot(stc._times,frontTimeCourses[0:2,:].T)
    #         ax5.set_ylim([ymin, ymax])
    #         ax5.legend([frontLabels[0].name, frontLabels[1].name])

    #         ax6.plot(stc._times,postcTimeCourses[0:2,:].T)
    #         ax6.set_ylim([ymin, ymax])
    #         ax6.legend([postcLabels[0].name, postcLabels[1].name])

    #         ax7.plot(stc._times,occTimeCourses[0:2,:].T)
    #         ax7.set_ylim([ymin, ymax])
    #         ax7.legend([occLabels[0].name, occLabels[1].name])

    #         ax8.plot(stc._times,subcortTimeCourses.mean(axis=1).T)
    #         # ax8.plot(stc._times,subcortTimeCourses[:,2,:].T)
    #         # ax8.set_ylim([ymin, ymax])
    #         ax8.legend(labels_vol)

    audTimeCoursesAllSubjs = np.array(audTimeCoursesAllSubjs)
    tempTimeCoursesAllSubjs = np.array(tempTimeCoursesAllSubjs)
    supParTimeCoursesAllSubjs = np.array(supParTimeCoursesAllSubjs)
    infParTimeCoursesAllSubjs = np.array(infParTimeCoursesAllSubjs)
    frontTimeCoursesAllSubjs = np.array(frontTimeCoursesAllSubjs)
    postcTimeCoursesAllSubjs = np.array(postcTimeCoursesAllSubjs)
    occTimeCoursesAllSubjs = np.array(occTimeCoursesAllSubjs)
    subcortTimeCoursesAllSubjs = np.array(subcortTimeCoursesAllSubjs)
    allSourceTimeCoursesAllSubjs = np.array(allSourceTimeCoursesAllSubjs)

    audTimeCoursesAvg = np.mean(audTimeCoursesAllSubjs, axis=0)
    tempTimeCoursesAvg = np.mean(tempTimeCoursesAllSubjs, axis=0)
    supParTimeCoursesAvg = np.mean(supParTimeCoursesAllSubjs, axis=0)
    infParTimeCoursesAvg = np.mean(infParTimeCoursesAllSubjs, axis=0)
    frontTimeCoursesAvg = np.mean(frontTimeCoursesAllSubjs, axis=0)
    postcTimeCoursesAvg = np.mean(postcTimeCoursesAllSubjs, axis=0)
    occTimeCoursesAvg = np.mean(occTimeCoursesAllSubjs, axis=0)
    subcortTimeCoursesAvg = np.mean(subcortTimeCoursesAllSubjs, axis=0)
    allSourceTimeCoursesAvg = np.mean(allSourceTimeCoursesAllSubjs, axis=0)

    ymax = np.max([subcortTimeCoursesAvg.max(), allSourceTimeCoursesAvg.max()])
    ymin = np.min([subcortTimeCoursesAvg.min(), allSourceTimeCoursesAvg.min()])

    return (
        audTimeCoursesAvg,
        tempTimeCoursesAvg,
        supParTimeCoursesAvg,
        infParTimeCoursesAvg,
        frontTimeCoursesAvg,
        postcTimeCoursesAvg,
        occTimeCoursesAvg,
        subcortTimeCoursesAvg,
        allSourceTimeCoursesAvg,
        ymax,
        ymin,
        stc_vec._times,
    )


# %%

# %%
# nameOfRegressor = "_ANmodel_correctedLevels"
# lenResponse = 425
# fs = 5000

nameOfRegressor = "_ANmodel_maxFs"
lenResponse = 1393
fs = 16384

# nameOfRegressor = "~gammatone-1"
# lenResponse = 1100
# fs = 2000

# nameOfRegressor = "~gammatone-on-1"
# lenResponse = 1100
# fs = 2000

timeToAddToStart = 0
# timeToAddToStart = .008


doPC = True

nonVecMode = "mean_flip"
doVec = True


subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

subDirs = "/eegAndMeg/eeg/"

# This is everybody
subjectsToAverage = [
    "R3045",
    "R3089",
    "R3093",
    "R3095",
    "R3151",
    "R2774",
    "R3152",
    "R2877",
    "R3157",
    "R2783",
    "R3170",
    "R3172",
    "R3193",
    "R3184",
    "R3214",
]

# subjectsToAverage = [
#     "R3089",
#     "R3093",
#     "R3095",
#     "R3151",
#     "R2774",
#     "R3152",
#     "R2877",
#     "R3157",
#     "R2783",
#     "R3170",
#     "R3172",
#     "R3184",
#     "R3214",
# ]


# subjectsToAverage = [
#     "R3093",
#     "R3095",
#     "R3151",
#     "R3152",
#     "R2877",
#     "R3157",
#     "R2783",
# ]

# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877", "R3157"]
# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877"]
# subjectsToAverage=["R3095", "R3151", "R2774", "R3152", "R2877", "R3157", "R2783"]
# subjectsToAverage=["R2877", "R3151", "R3152"]
# subjectsToAverage=["R2877", "R3151"]
# subjectsToAverage=["R3151"]
# subjectsToAverage=["R2877"]
# subjectsToAverage=["R3152"]


# typeOfRegressor = "mix"
typeOfRegressors = ["mix", "target", "target", "distractor", "distractor"]
# typeOfRegressors = ["mix", "target", "target"]
# typeOfRegressor = "distractor"

filenameSuffixes = ["_quiet", "_easy", "_hard", "_easy", "_hard"]
# filenameSuffixes = ["_quiet", "_easy", "_hard"]
# filenameSuffix = "_easy"
# filenameSuffix = "_hard"
# filenameSuffixes = ""

allSourcesAllConds = dict()
condLabels = []

for iType, typeOfRegressor in enumerate(typeOfRegressors):

    filenameSuffix = filenameSuffixes[iType]

    # (
    #     audTimeCoursesAvg,
    #     tempTimeCoursesAvg,
    #     supParTimeCoursesAvg,
    #     infParTimeCoursesAvg,
    #     frontTimeCoursesAvg,
    #     postcTimeCoursesAvg,
    #     occTimeCoursesAvg,
    #     subcortTimeCoursesAvg,
    #     allSourceTimeCoursesAvg
    # ) = extractSourcesSingleCondition(subjectsToAverage, subDirs, nameOfRegressor, typeOfRegressor, filenameSuffix)

    condLabels.append(typeOfRegressor + filenameSuffix)

    allSourcesAllConds[condLabels[iType]] = extractSourcesSingleCondition(
        subjectsToAverage,
        subDirs,
        nameOfRegressor,
        typeOfRegressor,
        filenameSuffix,
        doPC,
        nonVecMode,
        doVec,
    )


# %%
from cycler import cycler


ymax = 0
ymin = 0

for condLabel in condLabels:

    ymax = np.max([ymax, allSourcesAllConds[condLabel][9]])
    ymin = np.min([ymin, allSourcesAllConds[condLabel][10]])


times = allSourcesAllConds[condLabels[0]][11]

figsize = [20, 7]

fig1, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(
    nrows=2, ncols=5, figsize=figsize
)

colors = ["blue", "cyan", "green", "red", "brown"]
custom_cycler = cycler(color=colors)
facecolor = "xkcd:light grey"

lenToRamp = 0.008
expToRamp = 1
rampFunction = np.concatenate(
    (
        np.linspace(0, 1, int(lenToRamp * fs)) ** expToRamp,
        np.ones(lenResponse - int(lenToRamp * fs)),
    )
)
rampFunction = rampFunction[:, None]


ax0.set_prop_cycle(custom_cycler)
# ax0.plot(times, np.stack((allSourcesAllConds[condLabels[0]][7][:,2], allSourcesAllConds[condLabels[1]][7][:,2],
#                                      allSourcesAllConds[condLabels[2]][7][:,2], allSourcesAllConds[condLabels[3]][7][:,2],
#                                      allSourcesAllConds[condLabels[4]][7][:,2]), axis=1))
ax0.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][7][:, 2] for i in range(len(condLabels))],
        axis=1,
    ),
)
# ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
ax0.set_ylim([ymin, ymax])
ax0.set_title("Brainstem")
ax0.set_facecolor(facecolor)
# ax0.legend(condLabels)

ax1.set_prop_cycle(custom_cycler)
# ax1.plot(times, np.stack((allSourcesAllConds[condLabels[0]][7][:,0], allSourcesAllConds[condLabels[1]][7][:,0],
#                                      allSourcesAllConds[condLabels[2]][7][:,0], allSourcesAllConds[condLabels[3]][7][:,0],
#                                      allSourcesAllConds[condLabels[4]][7][:,0]), axis=1))
ax1.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][7][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
# ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
ax1.set_ylim([ymin, ymax])
ax1.set_title("Left - Thalamus")
ax1.set_facecolor(facecolor)
# ax1.legend([audLabels[0].name, audLabels[1].name])

ax2.set_prop_cycle(custom_cycler)
# ax2.plot(times, np.stack((allSourcesAllConds[condLabels[0]][0][:,0], allSourcesAllConds[condLabels[1]][0][:,0],
#                                      allSourcesAllConds[condLabels[2]][0][:,0], allSourcesAllConds[condLabels[3]][0][:,0],
#                                      allSourcesAllConds[condLabels[4]][0][:,0]), axis=1))
ax2.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][0][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax2.set_ylim([ymin, ymax])
ax2.set_title("Left - Heschl's Gyrus")
ax2.set_facecolor(facecolor)
# ax2.legend([tempLabels[0].name, tempLabels[1].name])

ax3.set_prop_cycle(custom_cycler)
# ax3.plot(times, np.stack((allSourcesAllConds[condLabels[0]][1][:,0], allSourcesAllConds[condLabels[1]][1][:,0],
#                                      allSourcesAllConds[condLabels[2]][1][:,0], allSourcesAllConds[condLabels[3]][1][:,0],
#                                      allSourcesAllConds[condLabels[4]][1][:,0]), axis=1))
ax3.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][1][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax3.set_ylim([ymin, ymax])
ax3.set_title("Left - Superior Temporal Gyrus")
ax3.set_facecolor(facecolor)
# ax3.legend([supParLabels[0].name, supParLabels[1].name])

ax4.set_prop_cycle(custom_cycler)
# ax4.plot(times, np.stack((allSourcesAllConds[condLabels[0]][4][:,0], allSourcesAllConds[condLabels[1]][4][:,0],
#                                      allSourcesAllConds[condLabels[2]][4][:,0], allSourcesAllConds[condLabels[3]][4][:,0],
#                                      allSourcesAllConds[condLabels[4]][4][:,0]), axis=1))
ax4.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][6][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax4.set_ylim([ymin, ymax])
ax4.set_title("Left - Occipital Cortex Control")
ax4.set_facecolor(facecolor)


ax5.set_prop_cycle(custom_cycler)
# ax5.plot(stc._times, np.concatenate((allSourcesAllConds[condLabels[0]][7][:,2], allSourcesAllConds[condLabels[1]][7][:,2],
#                                      allSourcesAllConds[condLabels[2]][7][:,2], allSourcesAllConds[condLabels[3]][7][:,2],
#                                      allSourcesAllConds[condLabels[4]][7][:,2])))
ax5.plot(
    times,
    rampFunction
    * np.stack(
        [
            allSourcesAllConds[condLabels[i]][7][:, :2].mean(axis=1)
            for i in range(len(condLabels))
        ],
        axis=1,
    ),
)
# # ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
ax5.set_ylim([ymin, ymax])
ax5.set_title("Average Thalamus")
ax5.set_facecolor(facecolor)
# ax5.legend([audLabels[0].name, audLabels[1].name])

ax6.set_prop_cycle(custom_cycler)
# ax6.plot(times, np.stack((allSourcesAllConds[condLabels[0]][7][:,1], allSourcesAllConds[condLabels[1]][7][:,1],
#                                      allSourcesAllConds[condLabels[2]][7][:,1], allSourcesAllConds[condLabels[3]][7][:,1],
#                                      allSourcesAllConds[condLabels[4]][7][:,1]), axis=1))
ax6.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][7][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
# ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
ax6.set_ylim([ymin, ymax])
ax6.set_title("Right")
ax6.set_facecolor(facecolor)
# ax6.legend([audLabels[0].name, audLabels[1].name])

ax7.set_prop_cycle(custom_cycler)
# ax7.plot(times, np.stack((allSourcesAllConds[condLabels[0]][0][:,1], allSourcesAllConds[condLabels[1]][0][:,1],
#                                      allSourcesAllConds[condLabels[2]][0][:,1], allSourcesAllConds[condLabels[3]][0][:,1],
#                                      allSourcesAllConds[condLabels[4]][0][:,1]), axis=1))
ax7.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][0][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax7.set_ylim([ymin, ymax])
ax7.set_title("Right")
ax7.set_facecolor(facecolor)
# ax7.legend([tempLabels[0].name, tempLabels[1].name])


ax8.set_prop_cycle(custom_cycler)
# ax8.plot(times, np.stack((allSourcesAllConds[condLabels[0]][1][:,1], allSourcesAllConds[condLabels[1]][1][:,1],
#                                      allSourcesAllConds[condLabels[2]][1][:,1], allSourcesAllConds[condLabels[3]][1][:,1],
#                                      allSourcesAllConds[condLabels[4]][1][:,1]), axis=1))
ax8.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][1][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax8.set_ylim([ymin, ymax])
ax8.set_title("Right")
ax8.set_facecolor(facecolor)
# ax8.legend([supParLabels[0].name, supParLabels[1].name])

ax9.set_prop_cycle(custom_cycler)
# ax9.plot(times, np.stack((allSourcesAllConds[condLabels[0]][4][:,1], allSourcesAllConds[condLabels[1]][4][:,1],
#                                      allSourcesAllConds[condLabels[2]][4][:,1], allSourcesAllConds[condLabels[3]][4][:,1],
#                                      allSourcesAllConds[condLabels[4]][4][:,1]), axis=1))
ax9.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][6][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax9.set_ylim([ymin, ymax])
ax9.set_title("Right")
ax9.set_facecolor(facecolor)
ax9.legend(condLabels)

fig1.set_facecolor(facecolor)


# colors2 = ["green", "cyan", "blue"]
# custom_cycler2 = cycler(color=colors2)
# plt.figure()
# plt.gca().set_prop_cycle(custom_cycler2)
# plt.plot(times, rampFunction * np.stack((allSourcesAllConds[condLabels[i]][7][:,:2].mean(axis=1) for i in [2,1,0]), axis=1), linewidth=3)

# %%
# colors2 = ["red", "green", "blue"]
# colors2 = ["blue", "green", "red"]

# colors2 = ["green", "red", "blue"]
colors2 = ["cyan", "green", "red", "brown", "blue"]
custom_cycler2 = cycler(color=colors2)
plt.figure(figsize=[12, 5])
axAbr = plt.axes()
axAbr.set_prop_cycle(custom_cycler2)
for spine in axAbr.spines.values():
    spine.set_visible(False)
axAbr.axhline(0, color="black", linestyle="--")
axAbr.axvline(0, color="black", linestyle="--")
# plt.gca().set_facecolor(facecolor)
axAbr.plot(
    times,
    rampFunction
    * np.stack(
        [
            allSourcesAllConds[condLabels[i]][7][:, :2].mean(axis=1)
            for i in [1, 2, 3, 4, 0]
        ],
        axis=1,
    ),
    linewidth=3,
    alpha=0.75,
)
# axAbr.set_ylim([ymin * yScaleUp * scaleDown, ymax * yScaleUp * scaleDown])
axAbr.set_title("Thalamus and Brainstem", fontsize=16)
axAbr.set_xlabel("Time (sec)", fontsize=16)
axAbr.set_ylabel("Current (a.u.)", fontsize=16)
plt.tick_params("x", labelsize=16)
plt.tick_params("y", labelsize=16)
axAbr.autoscale(enable=True, axis="x", tight=True)
axAbr.grid()


# ymax = np.max([subcortTimeCoursesAvg.max(), allSourceTimeCoursesAvg.max()])
# ymin = np.min([subcortTimeCoursesAvg.min(), allSourceTimeCoursesAvg.min()])

# figsize = [20, 11]

# fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
#     nrows=2, ncols=4, figsize=figsize
# )


# ax1.plot(stc._times, audTimeCoursesAvg)
# # ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
# ax1.set_ylim([ymin, ymax])
# ax1.legend([audLabels[0].name, audLabels[1].name])

# ax2.plot(stc._times, tempTimeCoursesAvg)
# ax2.set_ylim([ymin, ymax])
# ax2.legend([tempLabels[0].name, tempLabels[1].name])

# ax3.plot(stc._times, supParTimeCoursesAvg)
# ax3.set_ylim([ymin, ymax])
# ax3.legend([supParLabels[0].name, supParLabels[1].name])

# ax4.plot(stc._times, infParTimeCoursesAvg)
# ax4.set_ylim([ymin, ymax])
# ax4.legend([infParLabels[0].name, infParLabels[1].name])

# ax5.plot(stc._times, frontTimeCoursesAvg)
# ax5.set_ylim([ymin, ymax])
# ax5.legend([frontLabels[0].name, frontLabels[1].name])

# ax6.plot(stc._times, postcTimeCoursesAvg)
# ax6.set_ylim([ymin, ymax])
# ax6.legend([postcLabels[0].name, postcLabels[1].name])

# ax7.plot(stc._times, occTimeCoursesAvg)
# ax7.set_ylim([ymin, ymax])
# ax7.legend([occLabels[0].name, occLabels[1].name])

# ax8.plot(stc._times, subcortTimeCoursesAvg)
# # ax8.plot(stc._times,subcortTimeCourses[:,1,:].T)
# # ax8.set_ylim([ymin, ymax])
# ax8.legend(labels_vol)

# %%

# %%
