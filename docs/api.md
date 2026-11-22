# Python API

This page documents the public entry point and the internal Python modules, using docstrings maintained alongside the source code. Most functions are **implementation-level interfaces** that expect specific GalCraft arrays and configuration structures; the stable way to run a simulation is the `GalCraft` command described in [Getting started](getting-started.md).

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

## Important implementation caveats

The current archived code includes a few utility branches that should be tested before direct API use: `Cot.gal2equ` (two-argument branch), `Cot.xyz2lbr` (three-argument branch), and the `interp_on_alpha` branch in `SpecGeneratorPix.setup_interpolator`. For normal command-line work, prefer the documented/tested interpolation mode and validate scientific output against the repository's test kit.
