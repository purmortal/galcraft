# GalCraft

**Build synthetic integral-field spectroscopy (IFS) data cubes from stellar-particle galaxy models.**

GalCraft converts stellar-particle catalogues and simple stellar population (SSP) spectral templates into mock observations, including optional line-of-sight velocities, extinction, instrumental broadening and observational noise. Output FITS cubes can be analysed using the same stellar-kinematic and stellar-population tools employed for observed IFS data.

GalCraft is described in [Wang et al. (2024), MNRAS, 534, 1175](https://doi.org/10.1093/mnras/stae2148). See also the [source code](https://github.com/purmortal/galcraft).

```{toctree}
:maxdepth: 2
:caption: Start here

getting-started
workflow
configuration
input-data
outputs
```

```{toctree}
:maxdepth: 2
:caption: Reference

methods
api
limitations
contributing
citation
```

:::{note}
This site documents the source provided in the `main` branch archive dated 30 December 2025. The companion [test-kit branch](https://github.com/purmortal/galcraft/tree/test_kit) contains example data and a configuration file. Large model files and SSP libraries are not bundled in the main repository.
:::
