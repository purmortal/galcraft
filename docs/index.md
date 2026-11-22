# GalCraft

**A Python framework for producing synthetic integral-field spectroscopy (IFS) data cubes from stellar-particle galaxy models.**

GalCraft combines the spatial positions, velocities, stellar population parameters and mass weights of simulated stars with simple stellar population (SSP) spectra. It creates mock IFS data products for comparison with observations, with options for line-of-sight Doppler shifts, dust attenuation, instrumental broadening and simplified observational noise.

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

GalCraft is described in [Wang et al. (2024), *MNRAS*, 534, 1175](https://doi.org/10.1093/mnras/stae2148). For usage or contributions, see the [GalCraft GitHub repository](https://github.com/purmortal/galcraft).

:::{note}
The instructions here reflect the archived GalCraft implementation used to prepare this documentation. The [test-kit branch](https://github.com/purmortal/galcraft/tree/test_kit) contains example inputs and configuration files; SSP libraries and galaxy models are not bundled with the main package.
:::
