# Tools and scripts for MEG and EEG analysis
## Setup and usage
- Many of the tools in this repository use auxiliary functions in the repository [neuroAndSignalTools](https://github.com/lerud/neuroAndSignalTools).
- Setting up a Python environment with `conda` or `mamba` along the lines of `environment.yml` in this repository will work, or `environment_cluster.yml` for minimal visualization and GUI components
## Associated materials
### Materials
- This code is used in the following manuscript:

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.04.730183-B31B1B)](https://doi.org/10.64898/2026.06.04.730183)
- This code is archived at the following location:

[![DOI](https://zenodo.org/badge/840058207.svg)](https://zenodo.org/badge/latestdoi/840058207)
- And the data associated with the manuscript, which can be analyzed with this code, is archived here:

[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23773280-1682D4)](https://doi.org/10.5281/zenodo.23773280)
### Instructions for using this code with the above data
- Constructing a Python environment with `environment.yml` will include all relevant packages for running this Python code. Using the specified Python version ensures that the associated pickled files can be unpickled and used.
- After changing the variable `parentDir` to where you untarred the files, the script `plotAverageTrfsMEEG.py` should load, plot, and analyze the sensor-space TRFs.
- The script `plotAbrMlrSources.py` should load the source-space TRFs and plot and analyze them, and the cluster test itself can either be run with that script, or loaded, by changing the variables `doTests` and `loadTests`.
