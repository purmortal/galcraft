# Welcome to GalCraft’s documentation!

**GalCraft** is a Python framework for generating synthetic integral-field spectroscopy (IFS) data cubes from Galactic chemical evolution (GCE) models and cosmological simulations.

GalCraft transforms stellar-particle distributions into mock IFS observations by assigning spectra to individual stellar particles based on their line-of-sight velocities and stellar population properties, including age, metallicity ([Fe/H]), and α-element abundance ([α/Fe]), using simple stellar population (SSP) spectral templates.

The resulting mock IFS data cubes can be analysed using the same techniques commonly applied to real IFS observations, including but not limited to:

- **Voronoi binning** ([Cappellari & Copin 2003](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C/abstract)).
- **Penalized Pixel-Fitting (pPXF)** ([Cappellari & Emsellem 2004](https://ui.adsabs.harvard.edu/abs/2004PASP..116..138C/abstract); [Cappellari 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract), [2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C/abstract)).
- **Absorption-line strength indices** (e.g., [Worthey 1994](https://ui.adsabs.harvard.edu/abs/1994ApJS...95..107W/abstract); [Schiavon 2007](https://ui.adsabs.harvard.edu/abs/2007ApJS..171..146S/abstract); [Thomas et al. 2011](https://ui.adsabs.harvard.edu/abs/2011MNRAS.412.2183T/abstract); [Martín-Navarro et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.3700M/abstract)).
- **Integrated IFS analysis pipelines**, such as GIST ([Bittner et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...628A.117B/abstract)) and nGIST ([Fraser-McKelvie et al. 2025](https://ui.adsabs.harvard.edu/abs/2025ascl.soft07015F/abstract)), which combine multiple analysis techniques.

By bridging stellar-particle simulations and observational IFS data, **GalCraft enables direct, like-for-like comparisons between models of the Milky Way or simulated galaxies and external galaxies observed with IFS instruments**, providing a framework for testing spectral analysis methods and investigating galaxy formation and evolution.


```{toctree}
:maxdepth: 2
:hidden:

getting-started
workflow
configuration
outputs
api
```

## Documentation

- **[Getting started](getting-started.md)** — install GalCraft, run the test kit and inspect a cube.
- **[Workflow](workflow.md)** — follow each processing stage and learn which modules implement it.
- **[Configuration](configuration.md)** — reference every supported JSON setting, directories and SSP/instrument inputs.
- **[Outputs](outputs.md)** — understand the FITS cubes, NumPy arrays, tables and diagnostic files.
- **[Python API](api.md)** — class and function reference generated directly from the source docstrings.

## Scientific reference

GalCraft is described in detail in [Wang et al. (2024), *MNRAS*, 534, 1175](https://doi.org/10.1093/mnras/stae2148). 

## GitLab Repository
You can access the source code of GalCraft and its previous releases directly in its official [GitHub repository](https://github.com/purmortal/galcraft). You are also welcome to open issues there.
