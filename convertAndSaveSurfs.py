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
import numpy as np
import scipy as sp
import mne
import eelbrain as eb
import shutil
import os
import glob

# %%
subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

subject = "mni"

# %%
# Prepare the obj files to open in Blender
surfs = glob.glob(f"{subjects_dir}/{subject}/bem/*surf")

if not os.path.exists(f"{subjects_dir}/{subject}/bem/ORIG"):
    os.mkdir(f"{subjects_dir}/{subject}/bem/ORIG")
    for surf in surfs:
        dirName = os.path.dirname(surf)
        baseName = os.path.basename(surf)
        shutil.copy(surf, f"{dirName}/ORIG/{baseName}")
else:
    print("ORIG directory already exists, so didn't create it or copy anything")

for surf in surfs:
    coords, faces = mne.read_surface(surf)
    mne.write_surface(surf[:-4] + "obj", coords, faces)

# %%
# Now, after having fixed the obj files, convert them back to surf files
for surf in surfs:
    coords, faces = mne.read_surface(surf[:-4] + "obj")
    mne.write_surface(surf[:-4] + "surf", coords, faces, overwrite=True)

# %%
