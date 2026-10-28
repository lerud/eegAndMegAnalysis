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
import numpy as np
import eelbrain as eb
import scipy as sp
import mat73
import matplotlib
import matplotlib.pyplot as plt
import os.path as op
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

# matplotlib.use('TkAgg')

# %%
# subjects_dir='/usr/local/freesurfer/subjects/'
# subject='JZS_OLD'
# subject='fsaverage'
# subject='MP2'

# %%
# plot_bem_kwargs = dict(
#    subject=subject,
#    subjects_dir=subjects_dir,
#    brain_surfaces=["pial","white"],
#    orientation="coronal",
#    slices=[50, 75, 100, 125, 150, 175, 200],
# )

# mne.viz.plot_bem(**plot_bem_kwargs)

# %%
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

# %%
tonesDenoised = mne.io.read_raw("tonesDenoised.fif")
events = mne.find_events(tonesDenoised)
tonesDenoised.set_eeg_reference(projection=True)  # needed for inverse modeling
mne.viz.plot_alignment(
    tonesDenoised.info,
    src=src,
    eeg=["original", "projected"],
    trans=trans,
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)

# %%
tonesDenoised.apply_proj()

# %%
fwd = mne.make_forward_solution(
    tonesDenoised.info,
    trans=trans,
    src=src,
    bem=bem,
    eeg=True,
    mindist=5.0,
    n_jobs=None,
)
print(fwd)

# %%
# toneEpochs=eb.load.unpickle('toneEpochs')
# tonesEvoked=eb.load.unpickle('tonesEvoked')

# %%
event_dict = {"Tones": 130}
reject_criteria = dict(
    #    mag=4000e-15,  # 4000 fT
    #   grad=4000e-13,  # 4000 fT/cm
    eeg=550e-6,  # 150 µV
    #    eog=250e-6,
)  # 250 µV
epochs = mne.Epochs(
    tonesDenoised,
    events,
    event_id=event_dict,
    tmin=-0.3,
    tmax=1,
    reject=reject_criteria,
    preload=True,
)
toneEpochs = epochs["Tones"]
tonesEvoked = toneEpochs.average()

# %%
noise_cov = mne.compute_covariance(
    toneEpochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, tonesDenoised.info)

# %%
inverse_operator = mne.minimum_norm.make_inverse_operator(
    tonesEvoked.info, fwd, noise_cov, loose=0.2, depth=0.8
)

# %%
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2
stc, residual = mne.minimum_norm.apply_inverse(
    tonesEvoked,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)

# %%
# fig, ax = plt.subplots()
# ax.plot(1e3 * stc.times, stc.data[::100, :].T)
# ax.set(xlabel="time (ms)", ylabel="%s value" % method)

# fig, ax = plt.subplots(1, 1)
# tonesEvoked.plot(axes=ax)
# for text in list(ax.texts):
#    text.remove()
# for line in ax.lines:
#    line.set_color("#98df81")
# residual.plot(axes=ax)

# %%
vertno_max, time_max = stc.get_peak(hemi="rh")

# subjects_dir = data_path / "subjects"
surfer_kwargs = dict(
    hemi="rh",
    subjects_dir=subjects_dir,
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    initial_time=time_max,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=10,
)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(
    vertno_max,
    coords_as_verts=True,
    hemi="rh",
    color="blue",
    scale_factor=0.6,
    alpha=0.5,
)
brain.add_text(
    0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
)

# %%
