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
def extractSourcesSingleCondition(
    subjectsToAverage,
    subDirs,
    nameOfRegressor,
    typeOfRegressor,
    filenameSuffix,
    doPC,
    vecMode,
    nonVecMode,
    doVec,
    subjects_dir,
):

    # filenamePrefix = "sources"
    filenamePrefix = "concurSources"

    transvGyrusTimeCoursesAllSubjs = []
    transvSulcusTimeCoursesAllSubjs = []
    planTempTimeCoursesAllSubjs = []
    insulaTimeCoursesAllSubjs = []
    planPolarTimeCoursesAllSubjs = []
    supTempGyrusTimeCoursesAllSubjs = []
    supTempSulcusTimeCoursesAllSubjs = []
    infFrontTimeCoursesAllSubjs = []
    midFrontTimeCoursesAllSubjs = []
    parahipTimeCoursesAllSubjs = []
    subcortTimeCoursesAllSubjs = []
    allSourceTimeCoursesAllSubjs = []

    for i, subject in enumerate(subjectsToAverage):

        # eegLocation = "/Users/karl/map/" + subject + subDirs
        megLocation = "/Volumes/Seagate/map/" + subject + subDirs

        (
            subject,
            mriSubject,
            labels_vol,
            src_vol,
            fwd_vol,
            inverse_operator_vol,
            stc_vec,
        ) = eb.load.unpickle(
            f"{megLocation}{filenamePrefix}{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle"
        )

        volLocation = f"{subjects_dir}/{mriSubject}/mri/aparc.a2009s+aseg.mgz"

        if doVec:

            allSourceTimeCourses = []

            labels = ["ctx_lh_G_temp_sup-G_T_transv", "ctx_rh_G_temp_sup-G_T_transv"]
            transvGyrusTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(transvGyrusTimeCourses.shape)
            allSourceTimeCourses.append(transvGyrusTimeCourses)

            labels = ["ctx_lh_S_temporal_transverse", "ctx_rh_S_temporal_transverse"]
            transvSulcusTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(transvSulcusTimeCourses.shape)
            allSourceTimeCourses.append(transvSulcusTimeCourses)

            labels = ["ctx_lh_G_temp_sup-Plan_tempo", "ctx_rh_G_temp_sup-Plan_tempo"]
            planTempTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(planTempTimeCourses.shape)
            allSourceTimeCourses.append(planTempTimeCourses)

            labels = ["ctx_lh_S_circular_insula_inf", "ctx_rh_S_circular_insula_inf"]
            insulaTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(insulaTimeCourses.shape)
            allSourceTimeCourses.append(insulaTimeCourses)

            labels = ["ctx_lh_G_temp_sup-Plan_polar", "ctx_rh_G_temp_sup-Plan_polar"]
            planPolarTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(planPolarTimeCourses.shape)
            allSourceTimeCourses.append(planPolarTimeCourses)

            labels = ["ctx_lh_G_temp_sup-Lateral", "ctx_rh_G_temp_sup-Lateral"]
            supTempGyrusTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(supTempGyrusTimeCourses.shape)
            allSourceTimeCourses.append(supTempGyrusTimeCourses)

            labels = ["ctx_lh_S_temporal_sup", "ctx_rh_S_temporal_sup"]
            supTempSulcusTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(supTempSulcusTimeCourses.shape)
            allSourceTimeCourses.append(supTempSulcusTimeCourses)

            labels = ["ctx_lh_G_front_inf-Triangul", "ctx_rh_G_front_inf-Triangul"]
            infFrontTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(infFrontTimeCourses.shape)
            allSourceTimeCourses.append(infFrontTimeCourses)

            labels = ["ctx_lh_G_front_middle", "ctx_rh_G_front_middle"]
            midFrontTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(midFrontTimeCourses.shape)
            allSourceTimeCourses.append(midFrontTimeCourses)

            labels = ["ctx_lh_G_oc-temp_med-Parahip", "ctx_rh_G_oc-temp_med-Parahip"]
            parahipTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:2, :, :],
                doPC,
            )
            print(parahipTimeCourses.shape)
            allSourceTimeCourses.append(parahipTimeCourses)

            labels = ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Brain-Stem"]
            subcortTimeCourses = calcPC(
                stc_vec.extract_label_time_course(
                    [volLocation, labels], src_vol, mode=vecMode
                )[0:3, :, :],
                doPC,
            )
            print(subcortTimeCourses.shape)

            allSourceTimeCourses = np.array(allSourceTimeCourses)

            transvGyrusTimeCoursesAllSubjs.append(transvGyrusTimeCourses)
            transvSulcusTimeCoursesAllSubjs.append(transvSulcusTimeCourses)
            planTempTimeCoursesAllSubjs.append(planTempTimeCourses)
            insulaTimeCoursesAllSubjs.append(insulaTimeCourses)
            planPolarTimeCoursesAllSubjs.append(planPolarTimeCourses)
            supTempGyrusTimeCoursesAllSubjs.append(supTempGyrusTimeCourses)
            supTempSulcusTimeCoursesAllSubjs.append(supTempSulcusTimeCourses)
            infFrontTimeCoursesAllSubjs.append(infFrontTimeCourses)
            midFrontTimeCoursesAllSubjs.append(midFrontTimeCourses)
            parahipTimeCoursesAllSubjs.append(parahipTimeCourses)
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

    transvGyrusTimeCoursesAllSubjs = np.array(transvGyrusTimeCoursesAllSubjs)
    transvSulcusTimeCoursesAllSubjs = np.array(transvSulcusTimeCoursesAllSubjs)
    planTempTimeCoursesAllSubjs = np.array(planTempTimeCoursesAllSubjs)
    insulaTimeCoursesAllSubjs = np.array(insulaTimeCoursesAllSubjs)
    planPolarTimeCoursesAllSubjs = np.array(planPolarTimeCoursesAllSubjs)
    supTempGyrusTimeCoursesAllSubjs = np.array(supTempGyrusTimeCoursesAllSubjs)
    supTempSulcusTimeCoursesAllSubjs = np.array(supTempSulcusTimeCoursesAllSubjs)
    infFrontTimeCoursesAllSubjs = np.array(infFrontTimeCoursesAllSubjs)
    midFrontTimeCoursesAllSubjs = np.array(midFrontTimeCoursesAllSubjs)
    parahipTimeCoursesAllSubjs = np.array(parahipTimeCoursesAllSubjs)
    subcortTimeCoursesAllSubjs = np.array(subcortTimeCoursesAllSubjs)
    allSourceTimeCoursesAllSubjs = np.array(allSourceTimeCoursesAllSubjs)

    transvGyrusTimeCoursesAvg = np.mean(transvGyrusTimeCoursesAllSubjs, axis=0)
    transvSulcusTimeCoursesAvg = np.mean(transvSulcusTimeCoursesAllSubjs, axis=0)
    planTempTimeCoursesAvg = np.mean(planTempTimeCoursesAllSubjs, axis=0)
    insulaTimeCoursesAvg = np.mean(insulaTimeCoursesAllSubjs, axis=0)
    planPolarTimeCoursesAvg = np.mean(planPolarTimeCoursesAllSubjs, axis=0)
    supTempGyrusTimeCoursesAvg = np.mean(supTempGyrusTimeCoursesAllSubjs, axis=0)
    supTempSulcusTimeCoursesAvg = np.mean(supTempSulcusTimeCoursesAllSubjs, axis=0)
    infFrontTimeCoursesAvg = np.mean(infFrontTimeCoursesAllSubjs, axis=0)
    midFrontTimeCoursesAvg = np.mean(midFrontTimeCoursesAllSubjs, axis=0)
    parahipTimeCoursesAvg = np.mean(parahipTimeCoursesAllSubjs, axis=0)
    subcortTimeCoursesAvg = np.mean(subcortTimeCoursesAllSubjs, axis=0)
    allSourceTimeCoursesAvg = np.mean(allSourceTimeCoursesAllSubjs, axis=0)

    ymax = np.max([subcortTimeCoursesAvg.max(), allSourceTimeCoursesAvg.max()])
    ymin = np.min([subcortTimeCoursesAvg.min(), allSourceTimeCoursesAvg.min()])

    return (
        transvGyrusTimeCoursesAvg,
        transvSulcusTimeCoursesAvg,
        planTempTimeCoursesAvg,
        insulaTimeCoursesAvg,
        planPolarTimeCoursesAvg,
        supTempGyrusTimeCoursesAvg,
        supTempSulcusTimeCoursesAvg,
        infFrontTimeCoursesAvg,
        midFrontTimeCoursesAvg,
        parahipTimeCoursesAvg,
        subcortTimeCoursesAvg,
        allSourceTimeCoursesAvg,
        ymax,
        ymin,
        stc_vec._times,
    )


# %%
# nameOfRegressor = "_ANmodel_correctedLevels"
# lenResponse = 425
# fs = 5000


lenResponse = 426
fs = 500


# nameOfRegressor = "~gammatone-1"

# nameOfRegressor = "~gammatone-on-1"

# nameOfRegressor = "~wordOnsets_gaussian15msSD"

# nameOfRegressor = "~phoneOnsets_gaussian15msSD"

# nameOfRegressor = "~phsurp"

nameOfRegressor = "~cohtent"

# nameOfRegressor = "~wordprob"

# nameOfRegressor = "~wordsurp"

timeToAddToStart = 0
# timeToAddToStart = .008


doPC = True

vecMode = "mean"
nonVecMode = "mean_flip"
doVec = True


subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

subDirs = "/eegAndMeg/meg/"


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
#     "R3093",
#     "R3095",
#     "R3151",
#     "R3152",
#     "R2877",
#     "R3157",
#     "R2783",
# ]
# subjectsToAverage=["R3045"]


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

    condLabels.append(typeOfRegressor + filenameSuffix)

    allSourcesAllConds[condLabels[iType]] = extractSourcesSingleCondition(
        subjectsToAverage,
        subDirs,
        nameOfRegressor,
        typeOfRegressor,
        filenameSuffix,
        doPC,
        vecMode,
        nonVecMode,
        doVec,
        subjects_dir,
    )


# %%
from cycler import cycler


ymax = 0
ymin = 0

for condLabel in condLabels:

    ymax = np.max([ymax, allSourcesAllConds[condLabel][12]])
    ymin = np.min([ymin, allSourcesAllConds[condLabel][13]])


times = allSourcesAllConds[condLabels[0]][14]

figsize = [24, 14]

fig1, (
    (ax0, ax1, ax2, ax3, ax4, ax5),
    (ax6, ax7, ax8, ax9, ax10, ax11),
    (ax12, ax13, ax14, ax15, ax16, ax17),
    (ax18, ax19, ax20, ax21, ax22, ax23),
) = plt.subplots(nrows=4, ncols=6, figsize=figsize)

colors = ["blue", "cyan", "green", "red", "brown"]
custom_cycler = cycler(color=colors)
facecolor = "xkcd:light grey"

# lenToRamp = 0.008
lenToRamp = 0
expToRamp = 1
rampFunction = np.concatenate(
    (
        np.linspace(0, 1, int(lenToRamp * fs)) ** expToRamp,
        np.ones(lenResponse - int(lenToRamp * fs)),
    )
)
rampFunction = rampFunction[:, None]


ax0.set_prop_cycle(custom_cycler)
ax0.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][10][:, 2] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax0.set_ylim([ymin, ymax])
ax0.set_title("Brainstem")
ax0.set_facecolor(facecolor)
# ax0.legend(condLabels)

ax1.set_prop_cycle(custom_cycler)
ax1.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][10][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax1.set_ylim([ymin, ymax])
ax1.set_title("Left - Thalamus")
ax1.set_facecolor(facecolor)
# ax1.legend([audLabels[0].name, audLabels[1].name])

ax2.set_prop_cycle(custom_cycler)
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


indivFigSize = [9, 5]
scaleDown = 1 / 1e9
yScaleUp = 1.017


figLeft = plt.figure(figsize=indivFigSize)
axLeft = plt.axes()
axLeft.set_prop_cycle(custom_cycler)
for spine in axLeft.spines.values():
    spine.set_visible(False)
axLeft.axhline(0, color="black", linestyle="--")
axLeft.axvline(0, color="black", linestyle="--")
axLeft.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][0][:, 0] for i in range(len(condLabels))],
        axis=1,
    )
    * scaleDown,
    linewidth=3,
)
axLeft.set_ylim([ymin * yScaleUp * scaleDown, ymax * yScaleUp * scaleDown])
axLeft.set_title("Left - Heschl's Gyrus", fontsize=16)
axLeft.set_xlabel("Time (sec)", fontsize=16)
axLeft.set_ylabel("Current (a.u.)", fontsize=16)
plt.tick_params("x", labelsize=16)
plt.tick_params("y", labelsize=16)
axLeft.autoscale(enable=True, axis="x", tight=True)
axLeft.grid()


ax3.set_prop_cycle(custom_cycler)
ax3.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][1][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax3.set_ylim([ymin, ymax])
ax3.set_title("Left - Heschl's sulcus")
ax3.set_facecolor(facecolor)
# ax3.legend([supParLabels[0].name, supParLabels[1].name])

ax4.set_prop_cycle(custom_cycler)
ax4.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][2][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax4.set_ylim([ymin, ymax])
ax4.set_title("Left - Planum temporale")
ax4.set_facecolor(facecolor)


ax5.set_prop_cycle(custom_cycler)
ax5.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][3][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
# # ax1.plot(stc._times,audTimeCourses[0:2,2,:].T)
ax5.set_ylim([ymin, ymax])
ax5.set_title("Left - Circular insula")
ax5.set_facecolor(facecolor)
ax5.legend(condLabels)

ax6.set_prop_cycle(custom_cycler)
ax6.plot(
    times,
    rampFunction
    * np.stack(
        [
            allSourcesAllConds[condLabels[i]][10][:, :2].mean(axis=1)
            for i in range(len(condLabels))
        ],
        axis=1,
    ),
)
ax6.set_ylim([ymin, ymax])
ax6.set_title("Average thalamus")
ax6.set_facecolor(facecolor)
# ax6.legend([audLabels[0].name, audLabels[1].name])

ax7.set_prop_cycle(custom_cycler)
ax7.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][10][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax7.set_ylim([ymin, ymax])
ax7.set_title("Right")
ax7.set_facecolor(facecolor)
# ax7.legend([tempLabels[0].name, tempLabels[1].name])


ax8.set_prop_cycle(custom_cycler)
ax8.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][0][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax8.set_ylim([ymin, ymax])
ax8.set_title("Right")
ax8.set_facecolor(facecolor)
# ax8.legend([supParLabels[0].name, supParLabels[1].name])


figRight = plt.figure(figsize=indivFigSize)
axRight = plt.axes()
axRight.set_prop_cycle(custom_cycler)
for spine in axRight.spines.values():
    spine.set_visible(False)
axRight.axhline(0, color="black", linestyle="--")
axRight.axvline(0, color="black", linestyle="--")
axRight.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][0][:, 1] for i in range(len(condLabels))],
        axis=1,
    )
    * scaleDown,
    linewidth=3,
)
axRight.set_ylim([ymin * yScaleUp * scaleDown, ymax * yScaleUp * scaleDown])
axRight.set_title("Right - Heschl's Gyrus", fontsize=16)
axRight.set_xlabel("Time (sec)", fontsize=16)
axRight.set_ylabel("Current (a.u.)", fontsize=16)
plt.tick_params("x", labelsize=16)
plt.tick_params("y", labelsize=16)
axRight.autoscale(enable=True, axis="x", tight=True)
axRight.grid()


ax9.set_prop_cycle(custom_cycler)
ax9.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][1][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax9.set_ylim([ymin, ymax])
ax9.set_title("Right")
ax9.set_facecolor(facecolor)
# ax9.legend(condLabels)

ax10.set_prop_cycle(custom_cycler)
ax10.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][2][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax10.set_ylim([ymin, ymax])
ax10.set_title("Right")
ax10.set_facecolor(facecolor)
# ax10.legend([supParLabels[0].name, supParLabels[1].name])

ax11.set_prop_cycle(custom_cycler)
ax11.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][3][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax11.set_ylim([ymin, ymax])
ax11.set_title("Right")
ax11.set_facecolor(facecolor)
# ax11.legend(condLabels)


ax12.set_prop_cycle(custom_cycler)
ax12.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][4][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax12.set_ylim([ymin, ymax])
ax12.set_title("Left - Planum polare")
ax12.set_facecolor(facecolor)
# ax12.legend(condLabels)

ax13.set_prop_cycle(custom_cycler)
ax13.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][5][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax13.set_ylim([ymin, ymax])
ax13.set_title("Left - Sup. temp. gyrus")
ax13.set_facecolor(facecolor)
# ax13.legend([audLabels[0].name, audLabels[1].name])

ax14.set_prop_cycle(custom_cycler)
ax14.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][6][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax14.set_ylim([ymin, ymax])
ax14.set_title("Left - Sup. temp. sulcus")
ax14.set_facecolor(facecolor)
# ax14.legend([tempLabels[0].name, tempLabels[1].name])

ax15.set_prop_cycle(custom_cycler)
ax15.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][7][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax15.set_ylim([ymin, ymax])
ax15.set_title("Left - Inf. frontal gyrus")
ax15.set_facecolor(facecolor)
# ax15.legend([supParLabels[0].name, supParLabels[1].name])

ax16.set_prop_cycle(custom_cycler)
ax16.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][8][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax16.set_ylim([ymin, ymax])
ax16.set_title("Left - Mid. frontal gyrus")
ax16.set_facecolor(facecolor)


ax17.set_prop_cycle(custom_cycler)
ax17.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][9][:, 0] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax17.set_ylim([ymin, ymax])
ax17.set_title("Left - Parahip. gyrus")
ax17.set_facecolor(facecolor)
# ax17.legend([audLabels[0].name, audLabels[1].name])

ax18.set_prop_cycle(custom_cycler)
ax18.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][4][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax18.set_ylim([ymin, ymax])
ax18.set_title("Right")
ax18.set_facecolor(facecolor)
# ax18.legend([audLabels[0].name, audLabels[1].name])

ax19.set_prop_cycle(custom_cycler)
ax19.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][5][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax19.set_ylim([ymin, ymax])
ax19.set_title("Right")
ax19.set_facecolor(facecolor)
# ax19.legend([tempLabels[0].name, tempLabels[1].name])


ax20.set_prop_cycle(custom_cycler)
ax20.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][6][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax20.set_ylim([ymin, ymax])
ax20.set_title("Right")
ax20.set_facecolor(facecolor)
# ax20.legend([supParLabels[0].name, supParLabels[1].name])

ax21.set_prop_cycle(custom_cycler)
ax21.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][7][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax21.set_ylim([ymin, ymax])
ax21.set_title("Right")
ax21.set_facecolor(facecolor)
# ax21.legend(condLabels)

ax22.set_prop_cycle(custom_cycler)
ax22.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][8][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax22.set_ylim([ymin, ymax])
ax22.set_title("Right")
ax22.set_facecolor(facecolor)
# ax22.legend([supParLabels[0].name, supParLabels[1].name])

ax23.set_prop_cycle(custom_cycler)
ax23.plot(
    times,
    rampFunction
    * np.stack(
        [allSourcesAllConds[condLabels[i]][9][:, 1] for i in range(len(condLabels))],
        axis=1,
    ),
)
ax23.set_ylim([ymin, ymax])
ax23.set_title("Right")
ax23.set_facecolor(facecolor)
# ax23.legend(condLabels)


fig1.set_facecolor(facecolor)


# %%
