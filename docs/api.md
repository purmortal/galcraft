# Python API reference

GalCraft is primarily driven through its `GalCraft` command-line entry point. The functions below are also accessible directly in Python, but many are implementation-oriented and expect precisely structured arrays and configuration dictionaries.

## Main workflow

```{automodule} GalCraft.MainProcess
:members:
```

## Spatial binning

```{automodule} GalCraft.modules.Binner
:members:
```

## SSP loading

```{automodule} GalCraft.modules.SSPLoader
:members:
```

## Pixel-level spectra

```{automodule} GalCraft.modules.SpecGenerator
:members:
```

```{automodule} GalCraft.modules.SpecGeneratorPix
:members:
```

## FITS cube writer

```{automodule} GalCraft.modules.CubeMaker
:members:
```

## Coordinate transformations

```{automodule} GalCraft.modules.Cot
:members:
```

## Scientific utilities

```{automodule} GalCraft.modules.utils
:members:
```

## SSP-reader implementations

```{automodule} GalCraft.ssp_utils.miles_util
:members:
```

```{automodule} GalCraft.ssp_utils.conroyinmiles_util
:members:
```

```{automodule} GalCraft.ssp_utils.pegasehr_util
:members:
```

```{automodule} GalCraft.ssp_utils.pegasehr_interp_util
:members:
```

```{automodule} GalCraft.ssp_utils.xshooter_util
:members:
```

```{automodule} GalCraft.ssp_utils.alphaMC_util
:members:
```
