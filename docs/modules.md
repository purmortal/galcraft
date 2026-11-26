# Modules

This page documents the public entry point and the internal Python modules, using docstrings maintained alongside the source code. Most functions are **implementation-level interfaces** that expect specific GalCraft arrays and configuration structures; the stable way to run a simulation is the `GalCraft` command described in [Getting started](usage/getting-started.md).

## Module overview

| Module | Responsibility |
| --- | --- |
| `GalCraft.MainProcess` | Parse CLI options and orchestrate the end-to-end processing run. |
| `GalCraft.modules.Cot` | Transform particle coordinates and velocities into observer-frame quantities. |
| `GalCraft.modules.SSPLoader` | Read SSP libraries, prepare population grids and oversampled templates, and determine spectral-resolution matching. |
| `GalCraft.modules.Binner` | Choose fields of view, assign particles to spaxels, and save counts/intermediate tables. |
| `GalCraft.modules.SpecGenerator` | Distribute spaxel calculations across Python workers and assemble outputs. |
| `GalCraft.modules.SpecGeneratorPix` | Assign SSP spectra to particles; account for mass/distance, velocities and extinction; stack and broaden spectra. |
| `GalCraft.modules.CubeMaker` | Write FITS data cubes, wavelength axes and optional noise extensions. |
| `GalCraft.modules.utils` | Shared helpers for directories, logging, dust attenuation, Doppler shifts, spectral processing and plots. |
| `GalCraft.ssp_utils.*` | Library-specific SSP readers, including MILES, PEGASE-HR, α-MC and additional formats. |
| `GalCraft.instrument/` | Instrument preset JSON and tabulated line-spread-function files. |
| `GalCraft._version` | Package version metadata. |

The following **Python API** is generated from source docstrings using Sphinx autodoc. These internal interfaces are less stable than the command-line entry point.

## Main entry point

```{automodule} GalCraft.MainProcess
:members:
```

## Core pipeline modules

### Observer geometry: `Cot`

```{automodule} GalCraft.modules.Cot
:members:
```

### SSP loading: `SSPLoader`

```{automodule} GalCraft.modules.SSPLoader
:members:
```

### Spatial binning: `Binner`

```{automodule} GalCraft.modules.Binner
:members:
```

### Parallel spectrum generation: `SpecGenerator`

```{automodule} GalCraft.modules.SpecGenerator
:members:
```

### Per-pixel synthesis: `SpecGeneratorPix`

```{automodule} GalCraft.modules.SpecGeneratorPix
:members:
```

### FITS cube output: `CubeMaker`

```{automodule} GalCraft.modules.CubeMaker
:members:
```

### Shared utilities: `utils`

```{automodule} GalCraft.modules.utils
:members:
```

## SSP library readers

### MILES

```{automodule} GalCraft.ssp_utils.miles_util
:members:
```

### Conroy in MILES

```{automodule} GalCraft.ssp_utils.conroyinmiles_util
:members:
```

### PEGASE-HR

```{automodule} GalCraft.ssp_utils.pegasehr_util
:members:
```

### Interpolated PEGASE-HR

```{automodule} GalCraft.ssp_utils.pegasehr_interp_util
:members:
```

### X-shooter

```{automodule} GalCraft.ssp_utils.xshooter_util
:members:
```

### Alpha-enhanced models

```{automodule} GalCraft.ssp_utils.alphaMC_util
:members:
```

