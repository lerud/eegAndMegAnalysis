# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: python3
# ---

# %%
import mne
import eelbrain as eb
import numpy as np
import scipy as sp
import matplotlib
import os
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import ipyniivue
from ipyniivue import NiiVue, SliceType
import ipywidgets as widgets
import nibabel as nib
import os
import subprocess
from IPython.display import Image
from IPython.display import display
import IPython.display
import nilearn
import nilearn.image as nimg
from nilearn.maskers import NiftiSpheresMasker
from nilearn import datasets

plt.ion()

# %%
startToPlot = 9.2
stopToPlot = 9.6
mainFs = 24000
t = np.linspace(startToPlot, stopToPlot, int((stopToPlot - startToPlot) * mainFs))
partialColor = 0.3
fullColor = 1
linewidthTimeseries = 0.5

speechFs, speechStim = sp.io.wavfile.read("/Volumes/Seagate/map/stimAndPredictors/mixes/A_mix6.wav")
speechStim = sp.stats.zscore(sp.signal.resample(speechStim, int(np.round(len(speechStim) * mainFs / speechFs))))
speechStimPlot = speechStim[int(np.round(startToPlot * mainFs)): int(np.round(startToPlot * mainFs)) + len(t)]

speechAN = eb.load.unpickle("/Volumes/Seagate/map/stimAndPredictors/mixes/predictors/A_mix6_ANmodel_fs5000-0.pickle")
speechAN = sp.stats.zscore(sp.signal.resample(speechAN, int(np.round(len(speechAN) * mainFs / 5000))))
speechANPlot = speechAN[int(np.round(startToPlot * mainFs)): int(np.round(startToPlot * mainFs)) + len(t)]

clickFs, clickStim  = sp.io.wavfile.read("regressorWork/clickABRstim.wav")
clickStim = sp.stats.zscore(sp.signal.resample(clickStim, int(np.round(len(clickStim) * mainFs / clickFs))))
clickStimPlot = clickStim[int(np.round(startToPlot * mainFs)): int(np.round(startToPlot * mainFs)) + len(t)]

clickAN = eb.load.unpickle("regressorWork/clickANmodelMeg.pickle").squeeze()
clickAN = sp.stats.zscore(sp.signal.resample(clickAN, int(np.round(len(clickAN) * mainFs / 5000))))
clickANPlot = clickAN[int(np.round(startToPlot * mainFs)): int(np.round(startToPlot * mainFs)) + len(t)]


norm_factor = np.sqrt(np.sum(np.abs(speechStim)**2) * np.sum(np.abs(speechStim)**2))
speechStimAuto = sp.signal.correlate(speechStim, speechStim, mode="full") / norm_factor
lags = sp.signal.correlation_lags(len(speechStim), len(speechStim), mode="full")
speechStimLags = lags * 1/mainFs * 1000  # Now the lags are in milliseconds

norm_factor = np.sqrt(np.sum(np.abs(speechAN)**2) * np.sum(np.abs(speechAN)**2))
speechANAuto = sp.signal.correlate(speechAN, speechAN, mode="full") / norm_factor
lags = sp.signal.correlation_lags(len(speechAN), len(speechAN), mode="full")
speechANLags = lags * 1/mainFs * 1000  # Now the lags are in milliseconds

norm_factor = np.sqrt(np.sum(np.abs(clickStim)**2) * np.sum(np.abs(clickStim)**2))
clickStimAuto = sp.signal.correlate(clickStim, clickStim, mode="full") / norm_factor
lags = sp.signal.correlation_lags(len(clickStim), len(clickStim), mode="full")
clickStimLags = lags * 1/mainFs * 1000  # Now the lags are in milliseconds

norm_factor = np.sqrt(np.sum(np.abs(clickAN)**2) * np.sum(np.abs(clickAN)**2))
clickANAuto = sp.signal.correlate(clickAN, clickAN, mode="full") / norm_factor
lags = sp.signal.correlation_lags(len(clickAN), len(clickAN), mode="full")
clickANLags = lags * 1/mainFs * 1000  # Now the lags are in milliseconds


# %%
blues = plt.colormaps['Blues_r']
oranges = plt.colormaps['Oranges_r']

stimFig, stimAxs = plt.subplots(2, 2, figsize=[10, 6])
stimAxs[0, 0].plot(t, speechStimPlot, color=blues(partialColor), linewidth=linewidthTimeseries)
stimAxs[0, 0].set_title("Speech mixture stimulus")
stimAxs[0, 0].set_xticklabels([])
stimAxs[0, 0].set_ylabel("Z-scored amplitude")

stimAxs[1, 0].plot(t, speechANPlot, color=blues(fullColor), linewidth=linewidthTimeseries)
stimAxs[1, 0].set_title("Speech AN model regressor")
stimAxs[1, 0].set_xlabel("Time (s)")
stimAxs[1, 0].set_ylabel("Z-scored spikes per second")

stimAxs[0, 1].plot(t, clickStimPlot, color=oranges(partialColor), linewidth=linewidthTimeseries)
stimAxs[0, 1].set_title("Click stimulus")
stimAxs[0, 1].set_yticklabels([])
stimAxs[0, 1].set_xticklabels([])
stimAxs[0, 1].tick_params(axis="y", left=False)

stimAxs[1, 1].plot(t, clickANPlot, color=oranges(fullColor), linewidth=linewidthTimeseries)
stimAxs[1, 1].set_title("Click AN model regressor")
stimAxs[1, 1].set_xlabel("Time (s)")
stimAxs[1, 1].set_yticklabels([])
stimAxs[1, 1].tick_params(axis="y", left=False)



# %%
# corrPlotTimes = [0, 750]  # Start and end of plot in ms
corrPlotTimes = [0, 50]  # Start and end of plot in ms
linewidth = 3
autoFig, autoAxs = plt.subplots(1, 2, figsize=[10, 4])

corrPlotInds = []
corrPlotInds.append(np.abs(corrPlotTimes[0] - speechStimLags).argmin())
corrPlotInds.append(np.abs(corrPlotTimes[1] - speechStimLags).argmin())
autoAxs[0].plot(speechStimLags[corrPlotInds[0]: corrPlotInds[1]], speechStimAuto[corrPlotInds[0]:corrPlotInds[1]], color=blues(partialColor), alpha=0.75, linewidth=linewidth)

corrPlotInds = []
corrPlotInds.append(np.abs(corrPlotTimes[0] - speechANLags).argmin())
corrPlotInds.append(np.abs(corrPlotTimes[1] - speechANLags).argmin())
autoAxs[0].plot(speechANLags[corrPlotInds[0]: corrPlotInds[1]], speechANAuto[corrPlotInds[0]:corrPlotInds[1]], color=blues(fullColor), alpha=0.75, linewidth=linewidth)
autoAxs[0].set_xlabel("Lags (ms)")
autoAxs[0].set_ylabel("Autocorrelation coefficient")
autoAxs[0].set_ylim([-.45, 1])
autoAxs[0].legend(["Speech audio", "Speech AN model"])
autoAxs[0].set_title("Speech autocorrelation")

corrPlotInds = []
corrPlotInds.append(np.abs(corrPlotTimes[0] - clickStimLags).argmin())
corrPlotInds.append(np.abs(corrPlotTimes[1] - clickStimLags).argmin())
autoAxs[1].plot(clickStimLags[corrPlotInds[0]: corrPlotInds[1]], clickStimAuto[corrPlotInds[0]:corrPlotInds[1]], color=oranges(partialColor), alpha=0.75, linewidth=linewidth)

corrPlotInds = []
corrPlotInds.append(np.abs(corrPlotTimes[0] - clickANLags).argmin())
corrPlotInds.append(np.abs(corrPlotTimes[1] - clickANLags).argmin())
autoAxs[1].plot(clickANLags[corrPlotInds[0]: corrPlotInds[1]], clickANAuto[corrPlotInds[0]:corrPlotInds[1]], color=oranges(fullColor), alpha=0.75, linewidth=linewidth)
autoAxs[1].set_xlabel("Lags (ms)")
autoAxs[1].set_ylim([-.45, 1])
autoAxs[1].legend(["Click audio", "Click AN model"])
autoAxs[1].set_title("Clicks autocorrelation")


# %%
subjects = ["R3265", "R3275", "R3247", "R3292", "R3293", "R3294", "R3303", "R3304", "R3306", "R3313", "R3317", "R3320"]
ic_ras_mm = []
ic_tkreg_mm = []
mg_ras_mm = []
mg_tkreg_mm = []
ac_ras_mm = []
ac_tkreg_mm = []

# R3265
ic_ras_mm.append(np.array([1.99, -6.37, 18.62]))
ic_tkreg_mm.append(np.array([1.00, -24.48, -22.16]))
mg_ras_mm.append(np.array([14.19, -2.76, 27.69]))
mg_tkreg_mm.append(np.array([13.20, -20.87, -13.08]))
ac_ras_mm.append(np.array([52.84, 7.19, 38.49]))
ac_tkreg_mm.append(np.array([51.86, -10.92, -2.28]))

# R3275
ic_ras_mm.append(np.array([2.66, -7.08, 15.61]))
ic_tkreg_mm.append(np.array([-0.72, -26.98, -26.36]))
mg_ras_mm.append(np.array([14.78, -1.23, 22.16]))
mg_tkreg_mm.append(np.array([11.41, -21.13, -19.81]))
ac_ras_mm.append(np.array([43.88, 3.30, 34.44]))
ac_tkreg_mm.append(np.array([40.51, -16.60, -7.53]))

# R3247
ic_ras_mm.append(np.array([3.62, -8.20, 30.75]))
ic_tkreg_mm.append(np.array([0.20, -28.69, -7.62]))
mg_ras_mm.append(np.array([16.51, 0.86, 36.14]))
mg_tkreg_mm.append(np.array([13.09, -19.63, -2.24]))
ac_ras_mm.append(np.array([43.93, 11.75, 47.03]))
ac_tkreg_mm.append(np.array([40.51, -8.74, 8.66]))

# R3292
ic_ras_mm.append(np.array([2.68, -4.70, 2.94]))
ic_tkreg_mm.append(np.array([1.09, -31.76, -16.95]))
mg_ras_mm.append(np.array([15.74, -0.35, 9.04]))
mg_tkreg_mm.append(np.array([14.14, -27.40, -10.85]))
ac_ras_mm.append(np.array([46.00, -0.47, 27.63]))
ac_tkreg_mm.append(np.array([44.41, -27.52, 7.74]))

# R3293
ic_ras_mm.append(np.array([6.71, -3.34, 11.96]))
ic_tkreg_mm.append(np.array([-1.42, -29.21, -23.44]))
mg_ras_mm.append(np.array([17.88, 2.37, 18.76]))
mg_tkreg_mm.append(np.array([9.75, -23.50, -16.64]))
ac_ras_mm.append(np.array([49.07, 14.96, 29.60]))
ac_tkreg_mm.append(np.array([40.94, -10.92, -5.80]))

# R3294
ic_ras_mm.append(np.array([3.72, -10.32, 19.84]))
ic_tkreg_mm.append(np.array([3.36, -32.60, -22.75]))
mg_ras_mm.append(np.array([15.61, -3.81, 27.13]))
mg_tkreg_mm.append(np.array([15.24, -26.09, -15.45]))
ac_ras_mm.append(np.array([47.73, 1.69, 42.54]))
ac_tkreg_mm.append(np.array([47.36, -20.59, -0.05]))

# R3303
ic_ras_mm.append(np.array([1.43, -13.74, 4.17]))
ic_tkreg_mm.append(np.array([-0.78, -33.04, -23.47]))
mg_ras_mm.append(np.array([13.68, -8.71, 12.37]))
mg_tkreg_mm.append(np.array([11.46, -28.01, -15.27]))
ac_ras_mm.append(np.array([42.79, -2.98, 26.67]))
ac_tkreg_mm.append(np.array([40.57, -22.28, -0.97]))

# R3304
ic_ras_mm.append(np.array([1.53, -15.48, 3.36]))
ic_tkreg_mm.append(np.array([-0.65, -33.93, -21.30]))
mg_ras_mm.append(np.array([14.03, -9.82, 10.67]))
mg_tkreg_mm.append(np.array([11.86, -28.27, -13.98]))
ac_ras_mm.append(np.array([43.18, -2.93, 25.00]))
ac_tkreg_mm.append(np.array([41.01, -21.39, 0.34]))

# R3306
ic_ras_mm.append(np.array([-0.18, -7.57, 18.14]))
ic_tkreg_mm.append(np.array([-2.41, -30.45, -19.03]))
mg_ras_mm.append(np.array([9.98, -1.91, 25.52]))
mg_tkreg_mm.append(np.array([7.75, -24.79, -11.66]))
ac_ras_mm.append(np.array([39.46, 2.07, 41.54]))
ac_tkreg_mm.append(np.array([37.24, -20.81, 4.36]))

# R3313
ic_ras_mm.append(np.array([0.85, -16.49, 31.46]))
ic_tkreg_mm.append(np.array([-0.75, -20.87, -6.92]))
mg_ras_mm.append(np.array([12.70, -9.08, 38.52]))
mg_tkreg_mm.append(np.array([11.10, -13.46, 0.14]))
ac_ras_mm.append(np.array([38.18, 0.45, 52.75]))
ac_tkreg_mm.append(np.array([36.58, -3.93, 14.37]))

# R3317
ic_ras_mm.append(np.array([-0.87, -15.47, 15.04]))
ic_tkreg_mm.append(np.array([-1.31, -25.19, -17.38]))
mg_ras_mm.append(np.array([11.23, -10.09, 23.75]))
mg_tkreg_mm.append(np.array([10.78, -19.82, -8.67]))
ac_ras_mm.append(np.array([40.95, -4.26, 40.69]))
ac_tkreg_mm.append(np.array([40.51, -13.98, 8.27]))

# R3320
ic_ras_mm.append(np.array([-0.14, -8.84, 19.66]))
ic_tkreg_mm.append(np.array([0.15, -22.17, -12.16]))
mg_ras_mm.append(np.array([12.51, -2.75, 26.56]))
mg_tkreg_mm.append(np.array([12.81, -16.08, -5.26]))
ac_ras_mm.append(np.array([39.67, 3.02, 40.09]))
ac_tkreg_mm.append(np.array([39.96, -10.31, 8.27]))

# %%
subjects_dir = os.path.expandvars("$SUBJECTS_DIR")
avgBrain = "mni"
bem_fs = eb.load.unpickle(f"{subjects_dir}/{avgBrain}/bem/conductorModel.pickle")
zooms = "auto"
depth = 0.8
volume_label = None
radius_mm = 4
src_fs = mne.setup_volume_source_space(
    avgBrain,
    mri=subjects_dir + f"/{avgBrain}/mri/aparc.a2009s+aseg.mgz",
    pos=8.0,
    bem=bem_fs,
    volume_label=volume_label,
    subjects_dir=subjects_dir,
    add_interpolator=True,  # just for speed, usually this should be True
    verbose=True,
)



parentDir = "/Volumes/Seagate/map/"

ic_spheres = []
mg_spheres = []
ac_spheres = []
ic_psfs = []
mg_psfs = []
ac_psfs = []

for iSubject, subject in enumerate(subjects):

    mriVol = subjects_dir + "/" + subject + "/mri/aparc.a2009s+aseg.mgz"
    bem = eb.load.unpickle(f"{subjects_dir}/{subject}/bem/conductorModel.pickle")
    trans = f"{parentDir}{subject}/eegAndMeg/eeg/eeg-trans.fif"
    noise_cov = eb.load.unpickle(f"{parentDir}{subject}/eegAndMeg/meg/highpassNoiseCovariance.pickle")
    data_cov = eb.load.unpickle(f"{parentDir}{subject}/eegAndMeg/meg/dataCovariance_fs5000.pickle")
    evoked = eb.load.unpickle(f"{parentDir}{subject}/eegAndMeg/meg/anMeegRecFieldAllFolds_target_easy_lambda0.pickle")[1][0]
    evoked.set_eeg_reference("average", projection=True)
    src = mne.setup_volume_source_space(
        subject,
        mri=mriVol,
        pos=8.0,
        bem=bem,
        volume_label=volume_label,
        subjects_dir=subjects_dir,
        add_interpolator=True,  # just for speed, usually this should be True
        verbose=True,
    )
    fwd = mne.make_forward_solution(
        evoked.info,
        trans,
        src,
        bem,
        mindist=5.0,  # ignore sources<=5mm from innerskull
        meg=True,
        eeg=True,
        n_jobs=None,
    )
    morph = mne.compute_source_morph(
        src=fwd["src"],
        subject_from=subject,
        subject_to=avgBrain,
        src_to=src_fs,
        subjects_dir=subjects_dir,
        niter_sdr=[5, 5, 2],
        niter_affine=[5, 5, 2],
        zooms=zooms,  # just for speed
        verbose=True,
    )
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, noise_cov, depth=depth, loose="auto", verbose=True
    )
    # snr = 3.0  # use smaller SNR for raw data
    snr = 9
    # snr = 20
    inv_method = "dSPM"  # sLORETA, MNE, dSPM
    resMat = mne.minimum_norm.make_inverse_resolution_matrix(
        fwd, inverse_operator, method=inv_method, lambda2=1.0 / snr**2
    )

    # weight_norm = "unit-noise-gain-invariant"
    # # weight_norm = "nai"
    # depthBeam = depth
    # reg = 0.05
    # lcmv = mne.beamformer.make_lcmv(info=evoked.info, 
    #                                 forward=fwd, 
    #                                 data_cov=data_cov, 
    #                                 reg=reg, 
    #                                 noise_cov=noise_cov, 
    #                                 label=None, 
    #                                 pick_ori=None, 
    #                                 rank="info", 
    #                                 weight_norm=weight_norm, 
    #                                 reduce_rank=False, 
    #                                 depth=depthBeam, 
    #                                 inversion="matrix", 
    #                                 verbose=None)
    # beamResMat = mne.beamformer.make_lcmv_resolution_matrix(lcmv, fwd, evoked.info)

    coords_m = src[0]["rr"]  # Coordinates in meters (FreeSurfer surface RAS)
    vertno = src[0]["vertno"]  # Active vertex numbers/indices in the source space
    valid_coords = coords_m[vertno]

    # 2. Build a KDTree for fast nearest-neighbor lookup
    tree = sp.spatial.KDTree(valid_coords)





    target_ras_m = ic_tkreg_mm[iSubject] / 1000

    # Query the tree
    dist, closest_idx = tree.query(target_ras_m)

    # Get the actual source space vertex ID (vertno) and array index
    source_space_index = closest_idx  # index within src[hemi_idx]['vertno']
    vertex_id = vertno[closest_idx]

    print(f"Closest vertex index in source space: {source_space_index}")
    print(f"FreeSurfer vertex ID: {vertex_id}")
    print(f"Distance to target: {dist * 1000:.2f} mm")
    sources = [source_space_index]
    # stc_psf = mne.minimum_norm.get_point_spread(resMat, fwd["src"], sources, norm=True, vector=True)
    stc_psf = mne.minimum_norm.get_point_spread(resMat, fwd["src"], sources, norm=True)
    verttrue_vol = fwd["src"][0]["vertno"][sources]
    # find vertex with maximum in PSF
    max_vert_idx, _ = np.unravel_index(stc_psf.data.argmax(), stc_psf.data.shape)
    vert_max_ctf_vol = src[0]["vertno"][[max_vert_idx]]
    # brain_psf = stc_psf.plot_3d(
    #     subject=subject,
    #     src=fwd["src"],
    #     views="ven",
    #     subjects_dir=subjects_dir,
    #     volume_options=dict(alpha=0.5),
    # )
    # brain_psf.add_foci(
    #     verttrue_vol,
    #     coords_as_verts=True,
    #     scale_factor=1.25,
    #     hemi="vol",
    #     color="black",
    #     alpha=0.7,
    # )

    result = subprocess.run(["subject2mni_coords", "-m", f"/Volumes/Seagate/map/{subject}/mri/simnibsParc/m2m_{subject}/", "-c", f"{ic_ras_mm[iSubject][0]}", f"{ic_ras_mm[iSubject][1]}",f"{ic_ras_mm[iSubject][2]}"], capture_output=True)
    lines = result.stdout.decode('utf-8').strip().split('\n')
    clean_numbers_string = lines[1].strip('[]')
    target_ras_mni = np.fromstring(clean_numbers_string, sep=' ')
    template = datasets.load_mni152_template()

    # 2. Define your MNI coordinate (X, Y, Z) and radius in millimeters
    radius_mm = radius_mm             # Sphere radius in mm

    # 3. Initialize the sphere masker
    # 'mask_img' ensures the sphere is restricted to actual brain space
    masker = NiftiSpheresMasker(seeds=[tuple(target_ras_mni)], radius=radius_mm, mask_img=template)

    # 4. Fit the masker to the template to build the spatial configuration
    masker.fit(template)

    # 5. Transform the mask back into a 3D NIfTI image matrix
    # This gives you a Nifti1Image object where the sphere contains 1s and everything else is 0
    ic_spheres.append(masker.inverse_transform([1]))

    target_affine = np.eye(3) * 1.5
    stc_psf_mni = morph.apply(stc_psf)
    img = stc_psf_mni.as_volume(src=src_fs)
                
    ic_psfs.append(nimg.resample_img(
        img, 
        target_affine=target_affine, 
        interpolation="continuous" # Use 'nearest' for label maps
    ))
    




    target_ras_m = mg_tkreg_mm[iSubject] / 1000

    # Query the tree
    dist, closest_idx = tree.query(target_ras_m)

    # Get the actual source space vertex ID (vertno) and array index
    source_space_index = closest_idx  # index within src[hemi_idx]['vertno']
    vertex_id = vertno[closest_idx]

    print(f"Closest vertex index in source space: {source_space_index}")
    print(f"FreeSurfer vertex ID: {vertex_id}")
    print(f"Distance to target: {dist * 1000:.2f} mm")
    sources = [source_space_index]
    # stc_psf = mne.minimum_norm.get_point_spread(resMat, fwd["src"], sources, norm=True, vector=True)
    stc_psf = mne.minimum_norm.get_point_spread(resMat, fwd["src"], sources, norm=True)
    verttrue_vol = fwd["src"][0]["vertno"][sources]
    # find vertex with maximum in PSF
    max_vert_idx, _ = np.unravel_index(stc_psf.data.argmax(), stc_psf.data.shape)
    vert_max_ctf_vol = src[0]["vertno"][[max_vert_idx]]
    # brain_psf = stc_psf.plot_3d(
    #     subject=subject,
    #     src=fwd["src"],
    #     views="ven",
    #     subjects_dir=subjects_dir,
    #     volume_options=dict(alpha=0.5),
    # )
    # brain_psf.add_foci(
    #     verttrue_vol,
    #     coords_as_verts=True,
    #     scale_factor=1.25,
    #     hemi="vol",
    #     color="black",
    #     alpha=0.7,
    # )

    result = subprocess.run(["subject2mni_coords", "-m", f"/Volumes/Seagate/map/{subject}/mri/simnibsParc/m2m_{subject}/", "-c", f"{mg_ras_mm[iSubject][0]}", f"{mg_ras_mm[iSubject][1]}",f"{mg_ras_mm[iSubject][2]}"], capture_output=True)
    lines = result.stdout.decode('utf-8').strip().split('\n')
    clean_numbers_string = lines[1].strip('[]')
    target_ras_mni = np.fromstring(clean_numbers_string, sep=' ')
    template = datasets.load_mni152_template()

    # 2. Define your MNI coordinate (X, Y, Z) and radius in millimeters
    radius_mm = radius_mm             # Sphere radius in mm

    # 3. Initialize the sphere masker
    # 'mask_img' ensures the sphere is restricted to actual brain space
    masker = NiftiSpheresMasker(seeds=[tuple(target_ras_mni)], radius=radius_mm, mask_img=template)

    # 4. Fit the masker to the template to build the spatial configuration
    masker.fit(template)

    # 5. Transform the mask back into a 3D NIfTI image matrix
    # This gives you a Nifti1Image object where the sphere contains 1s and everything else is 0
    mg_spheres.append(masker.inverse_transform([1]))

    target_affine = np.eye(3) * 1.5
    stc_psf_mni = morph.apply(stc_psf)
    img = stc_psf_mni.as_volume(src=src_fs)
                
    mg_psfs.append(nimg.resample_img(
        img, 
        target_affine=target_affine, 
        interpolation="continuous" # Use 'nearest' for label maps
    ))
    


    target_ras_m = ac_tkreg_mm[iSubject] / 1000

    # Query the tree
    dist, closest_idx = tree.query(target_ras_m)

    # Get the actual source space vertex ID (vertno) and array index
    source_space_index = closest_idx  # index within src[hemi_idx]['vertno']
    vertex_id = vertno[closest_idx]

    print(f"Closest vertex index in source space: {source_space_index}")
    print(f"FreeSurfer vertex ID: {vertex_id}")
    print(f"Distance to target: {dist * 1000:.2f} mm")
    sources = [source_space_index]
    # stc_psf = mne.minimum_norm.get_point_spread(resMat, fwd["src"], sources, norm=True, vector=True)
    stc_psf = mne.minimum_norm.get_point_spread(resMat, fwd["src"], sources, norm=True)
    verttrue_vol = fwd["src"][0]["vertno"][sources]
    # find vertex with maximum in PSF
    max_vert_idx, _ = np.unravel_index(stc_psf.data.argmax(), stc_psf.data.shape)
    vert_max_ctf_vol = src[0]["vertno"][[max_vert_idx]]
    # brain_psf = stc_psf.plot_3d(
    #     subject=subject,
    #     src=fwd["src"],
    #     views="ven",
    #     subjects_dir=subjects_dir,
    #     volume_options=dict(alpha=0.5),
    # )
    # brain_psf.add_foci(
    #     verttrue_vol,
    #     coords_as_verts=True,
    #     scale_factor=1.25,
    #     hemi="vol",
    #     color="black",
    #     alpha=0.7,
    # )

    result = subprocess.run(["subject2mni_coords", "-m", f"/Volumes/Seagate/map/{subject}/mri/simnibsParc/m2m_{subject}/", "-c", f"{ac_ras_mm[iSubject][0]}", f"{ac_ras_mm[iSubject][1]}",f"{ac_ras_mm[iSubject][2]}"], capture_output=True)
    lines = result.stdout.decode('utf-8').strip().split('\n')
    clean_numbers_string = lines[1].strip('[]')
    target_ras_mni = np.fromstring(clean_numbers_string, sep=' ')
    template = datasets.load_mni152_template()

    # 2. Define your MNI coordinate (X, Y, Z) and radius in millimeters
    radius_mm = radius_mm             # Sphere radius in mm

    # 3. Initialize the sphere masker
    # 'mask_img' ensures the sphere is restricted to actual brain space
    masker = NiftiSpheresMasker(seeds=[tuple(target_ras_mni)], radius=radius_mm, mask_img=template)

    # 4. Fit the masker to the template to build the spatial configuration
    masker.fit(template)

    # 5. Transform the mask back into a 3D NIfTI image matrix
    # This gives you a Nifti1Image object where the sphere contains 1s and everything else is 0
    ac_spheres.append(masker.inverse_transform([1]))

    target_affine = np.eye(3) * 1.5
    stc_psf_mni = morph.apply(stc_psf)
    img = stc_psf_mni.as_volume(src=src_fs)
                
    ac_psfs.append(nimg.resample_img(
        img, 
        target_affine=target_affine, 
        interpolation="continuous" # Use 'nearest' for label maps
    ))
    







    


# %%
ic_sphere_mats = []
mg_sphere_mats = []
ac_sphere_mats = []
ic_psf_mats = []
mg_psf_mats = []
ac_psf_mats = []

for iSubject, subject in enumerate(subjects):
    ic_sphere_mats.append(ic_spheres[iSubject].get_fdata())
    mg_sphere_mats.append(mg_spheres[iSubject].get_fdata())
    ac_sphere_mats.append(ac_spheres[iSubject].get_fdata())
    ic_psf_mats.append(ic_psfs[iSubject].get_fdata())
    mg_psf_mats.append(mg_psfs[iSubject].get_fdata())
    ac_psf_mats.append(ac_psfs[iSubject].get_fdata())

ic_sphere_img = nib.Nifti1Image(np.array(ic_sphere_mats).mean(axis=0), ic_spheres[0].affine, ic_spheres[0].header)
mg_sphere_img = nib.Nifti1Image(np.array(mg_sphere_mats).mean(axis=0), mg_spheres[0].affine, mg_spheres[0].header)
ac_sphere_img = nib.Nifti1Image(np.array(ac_sphere_mats).mean(axis=0), ac_spheres[0].affine, ac_spheres[0].header)
ic_psf_img = nib.Nifti1Image(np.array(ic_psf_mats).mean(axis=0), ic_psfs[0].affine, ic_psfs[0].header)
mg_psf_img = nib.Nifti1Image(np.array(mg_psf_mats).mean(axis=0), mg_psfs[0].affine, mg_psfs[0].header)
ac_psf_img = nib.Nifti1Image(np.array(ac_psf_mats).mean(axis=0), ac_psfs[0].affine, ac_psfs[0].header)

# 6. Save the sphere image using nibabel
nib.save(ic_sphere_img, "/Users/karl/ic_spherical_roi.nii.gz")
nib.save(mg_sphere_img, "/Users/karl/mg_spherical_roi.nii.gz")
nib.save(ac_sphere_img, "/Users/karl/ac_spherical_roi.nii.gz")


threshold = 0.34  # Have been using this one, because it is close to what MNE automatically applies as the min in its own plots
# threshold = 0.65

data = ic_psf_img.get_fdata()
data = np.mean(data, axis=3)
# data = data[:, :, :, 2]
data[data<threshold] = 0
new_img = nib.Nifti1Image(data, ic_psf_img.affine, ic_psf_img.header)
# Save the resulting upsampled image
nib.save(new_img, "/Users/karl/ic_pointSpread.nii.gz")

data = mg_psf_img.get_fdata()
data = np.mean(data, axis=3)
# data = data[:, :, :, 2]
data[data<threshold] = 0
new_img = nib.Nifti1Image(data, mg_psf_img.affine, mg_psf_img.header)
# Save the resulting upsampled image
nib.save(new_img, "/Users/karl/mg_pointSpread.nii.gz")

data = ac_psf_img.get_fdata()
data = np.mean(data, axis=3)
# data = data[:, :, :, 2]
data[data<threshold] = 0
new_img = nib.Nifti1Image(data, ac_psf_img.affine, ac_psf_img.header)
# Save the resulting upsampled image
nib.save(new_img, "/Users/karl/ac_pointSpread.nii.gz")


# %%
volumes = []

volumeDict = {
    "path": f"/Users/karl/speechClickNiis/mni152.nii.gz",
    "colormap": "gray",
    # "visible": True,
    "opacity": 1,
}
volumes.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/usr/local/fsl/data/atlases/HarvardOxford/all.nii.gz",
    # "colormap": "warm",
    # "colormap": "green",
    "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": 1e-16,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/Users/karl/ic_pointSpread.nii.gz",
    "colormap": "inferno",
    # "colormap": "green",
    # "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": threshold,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/Users/karl/ic_spherical_roi.nii.gz",
    "colormap": "green",
    # "colormap": "green",
    # "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": threshold,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes.append(volumeDict)

nv = NiiVue()
nv.load_volumes(volumes)
nv.set_slice_type(slice_type=SliceType.MULTIPLANAR)
# nv.set_slice_type(slice_type=SliceType.RENDER)
# nv.load_meshes(meshes)
# nv.add_volume(volumes[0])
# nv.add_volume(volumes[1])
# nv.add_volume(volumes[2])
# nv
# nv.volumes[1].colormap = "red"
# nv.volumes[2].colormap = "blue"
# nv.volumes[0].opacity = 0.4
     
nv.opts.show_3d_crosshair = True
nv.opts.crosshair_color = [1, 1, 1, 0.5]
nv.opts.crosshair_width = .75
nv.opts.back_color = [0, 0, 0, 1]
nv.opts.is_colorbar = True
nv.opts.view_mode_hot_key = "KeyN"
nv.set_clip_plane(20, 180, 180)
nv.opts.clip_plane_color = [0, 1, 1, 1]

nv.height = 800
display(nv)

nv2 = NiiVue()
nv2.load_volumes(volumes)
# nv2.set_slice_type(slice_type=SliceType.MULTIPLANAR)
nv2.set_slice_type(slice_type=SliceType.RENDER)
# nv.load_meshes(meshes)
# nv.add_volume(volumes[0])
# nv.add_volume(volumes[1])
# nv.add_volume(volumes[2])
# nv
     

# nv.volumes[1].colormap = "red"
# nv.volumes[2].colormap = "blue"
# nv.volumes[0].opacity = 0.4
     

nv2.opts.show_3d_crosshair = True
nv2.opts.crosshair_color = [1, 1, 1, 0.5]
nv2.opts.crosshair_width = .75
nv2.opts.back_color = [0, 0, 0, 1]
nv2.opts.is_colorbar = True
nv2.opts.view_mode_hot_key = "KeyN"
nv2.set_clip_plane(20, 180, 180)
nv2.opts.clip_plane_color = [0, 1, 1, 1]


nv2.height = 800
display(nv2)
# config = ipyniivue.config_options.ConfigOptions
# config.trust_cal_min_max = False

# %%
volumes2 = []

volumeDict = {
    "path": f"/Users/karl/speechClickNiis/mni152.nii.gz",
    "colormap": "gray",
    # "visible": True,
    "opacity": 1,
}
volumes2.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/usr/local/fsl/data/atlases/HarvardOxford/all.nii.gz",
    # "colormap": "warm",
    # "colormap": "green",
    "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": 1e-16,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes2.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/Users/karl/mg_pointSpread.nii.gz",
    "colormap": "inferno",
    # "colormap": "green",
    # "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": threshold,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes2.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/Users/karl/mg_spherical_roi.nii.gz",
    "colormap": "green",
    # "colormap": "green",
    # "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": threshold,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes2.append(volumeDict)

nv3 = NiiVue()
nv3.load_volumes(volumes2)
nv3.set_slice_type(slice_type=SliceType.MULTIPLANAR)
# nv3.set_slice_type(slice_type=SliceType.RENDER)
# nv3.load_meshes(meshes)
# nv3.add_volume(volumes[0])
# nv3.add_volume(volumes[1])
# nv3.add_volume(volumes[2])
# nv3
# nv3.volumes[1].colormap = "red"
# nv3.volumes[2].colormap = "blue"
# nv3.volumes[0].opacity = 0.4
     
nv3.opts.show_3d_crosshair = True
nv3.opts.crosshair_color = [1, 1, 1, 0.5]
nv3.opts.crosshair_width = .75
nv3.opts.back_color = [0, 0, 0, 1]
nv3.opts.is_colorbar = True
nv3.opts.view_mode_hot_key = "KeyN"
nv3.set_clip_plane(20, 180, 180)
nv3.opts.clip_plane_color = [0, 1, 1, 1]

nv3.height = 800
display(nv3)

nv4 = NiiVue()
nv4.load_volumes(volumes2)
# nv4.set_slice_type(slice_type=SliceType.MULTIPLANAR)
nv4.set_slice_type(slice_type=SliceType.RENDER)
# nv4.load_meshes(meshes)
# nv4.add_volume(volumes[0])
# nv4.add_volume(volumes[1])
# nv4.add_volume(volumes[2])
# nv4
     

# nv4.volumes[1].colormap = "red"
# nv4.volumes[2].colormap = "blue"
# nv4.volumes[0].opacity = 0.4
     

nv4.opts.show_3d_crosshair = True
nv4.opts.crosshair_color = [1, 1, 1, 0.5]
nv4.opts.crosshair_width = .75
nv4.opts.back_color = [0, 0, 0, 1]
nv4.opts.is_colorbar = True
nv4.opts.view_mode_hot_key = "KeyN"
nv4.set_clip_plane(20, 180, 180)
nv4.opts.clip_plane_color = [0, 1, 1, 1]
     

nv4.height = 800
display(nv4)
# config = ipyniivue.config_options.ConfigOptions
# config.trust_cal_min_max = False

# %%
volumes3 = []

volumeDict = {
    "path": f"/Users/karl/speechClickNiis/mni152.nii.gz",
    "colormap": "gray",
    # "visible": True,
    "opacity": 1,
}
volumes3.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/usr/local/fsl/data/atlases/HarvardOxford/all.nii.gz",
    # "colormap": "warm",
    # "colormap": "green",
    "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": 1e-16,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes3.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/Users/karl/ac_pointSpread.nii.gz",
    "colormap": "inferno",
    # "colormap": "green",
    # "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": threshold,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes3.append(volumeDict)

volumeDict = {
    # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    "path": f"/Users/karl/ac_spherical_roi.nii.gz",
    "colormap": "green",
    # "colormap": "green",
    # "colormap": "copper",
    # "visible": True,
    "opacity": 0.75,
    # "cal_min": np.min(average_data[average_data>0]),
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": np.max(average_data[average_data<0])
    # "cal_min": threshold,
    # "cal_max": np.max(average_data),
    # "cal_minNeg": np.min(average_data[average_data<0]),
    # "cal_maxNeg": -1e-16,
    # "frame_4d": 52,
}
volumes3.append(volumeDict)

nv5 = NiiVue()
nv5.load_volumes(volumes3)
nv5.set_slice_type(slice_type=SliceType.MULTIPLANAR)
# nv5.set_slice_type(slice_type=SliceType.RENDER)
# nv5.load_meshes(meshes)
# nv5.add_volume(volumes[0])
# nv5.add_volume(volumes[1])
# nv5.add_volume(volumes[2])
# nv5
# nv5.volumes[1].colormap = "red"
# nv5.volumes[2].colormap = "blue"
# nv5.volumes[0].opacity = 0.4
     
nv5.opts.show_3d_crosshair = True
nv5.opts.crosshair_color = [1, 1, 1, 0.5]
nv5.opts.crosshair_width = .75
nv5.opts.back_color = [0, 0, 0, 1]
nv5.opts.is_colorbar = True
nv5.opts.view_mode_hot_key = "KeyN"
nv5.set_clip_plane(20, 180, 180)
nv5.opts.clip_plane_color = [0, 1, 1, 1]

nv5.height = 800
display(nv5)

nv6 = NiiVue()
nv6.load_volumes(volumes3)
# nv6.set_slice_type(slice_type=SliceType.MULTIPLANAR)
nv6.set_slice_type(slice_type=SliceType.RENDER)
# nv6.load_meshes(meshes)
# nv6.add_volume(volumes[0])
# nv6.add_volume(volumes[1])
# nv6.add_volume(volumes[2])
# nv6
     

# nv6.volumes[1].colormap = "red"
# nv6.volumes[2].colormap = "blue"
# nv6.volumes[0].opacity = 0.4
     

nv6.opts.show_3d_crosshair = True
nv6.opts.crosshair_color = [1, 1, 1, 0.5]
nv6.opts.crosshair_width = .75
nv6.opts.back_color = [0, 0, 0, 1]
nv6.opts.is_colorbar = True
nv6.opts.view_mode_hot_key = "KeyN"
nv6.set_clip_plane(20, 180, 180)
nv6.opts.clip_plane_color = [0, 1, 1, 1]
     

nv6.height = 800
display(nv6)
# config = ipyniivue.config_options.ConfigOptions
# config.trust_cal_min_max = False
