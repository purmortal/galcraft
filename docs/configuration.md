# Configuration reference

GalCraft uses two configuration layers:

1. A directory file, passed as `--default-dir`, with paths to inputs and outputs.
2. A JSON file named `setup_<config>.json`, with top-level dictionaries `oparams`, `ssp_params`, `cube_params` and `other_params`.

The [official test-kit setup JSON](https://raw.githubusercontent.com/purmortal/galcraft/test_kit/tests/configFiles/setup_test_kit.json) is the most reliable starting point for a complete working example. The parameters below are derived from both that setup and the code in `main`.

## Directory file

The parser in `utils.initialize_Dirs` accepts records such as:

```ini
configDir = /absolute/path/to/configFiles/
modelDir = /absolute/path/to/models/
templateDir = /absolute/path/to/templates/
outputDir = /absolute/path/to/output/
```

These four directories must exist. **Include the trailing slash**: parts of the current code concatenate directory paths as strings instead of using `os.path.join`.

## Observer parameters: `oparams`

| Key | Meaning |
| --- | --- |
| `distance` | Heliocentric distance of the placed galaxy, in kpc. |
| `l`, `b` | Galactic longitude and latitude of the model centre, in degrees. |
| `theta_zx`, `theta_yz` | Rotation angles, in degrees, used in `Cot.gc2helio`. |

The observer transformation is skipped if the input catalogue already contains `vr`; in that case the code also overwrites observer options with `null` in the in-memory configuration.

## Spectral templates: `ssp_params`

| Key | Meaning |
| --- | --- |
| `model` | SSP reader name, e.g. `miles`; must match a `<name>_util.py` file in `GalCraft/ssp_utils`. |
| `library` | Template-file glob, relative to `templateDir`. |
| `factor` | Integer oversampling factor for SSP spectra. |
| `FWHM_gal` | Target instrumental line-spread function: constant FWHM in Å or an instrument-directory filename. |
| `FWHM_tem` | Intrinsic SSP template FWHM in Å or a file supported by the relevant reader. |
| `dlam` | Target linear wavelength increment in Å, or `null` to retain the template grid. |
| `age_range`, `metal_range`, `alpha_range` | Optional two-element parameter limits or `null` for all available templates. |
| `wave_range` | Optional wavelength range `[min, max]` in Å for final cube output. |
| `spec_interpolator` | `nearest`, `linear`, or `interp_on_alpha` (the last requires multiple alpha-grid values and a special handling path). |

The code refuses an output LSF that is narrower than the template LSF everywhere, and drops wavelength samples where the requested broadening would be undefined. Do not assume extrapolation outside the supplied SSP parameter grid.

## Spatial cubes and physical effects: `cube_params`

| Key | Meaning |
| --- | --- |
| `x_coord`, `y_coord` | Coordinate columns for binning, generally `l`/`b` or `ra`/`dec`. |
| `instrument` | `DIY`, `DEFAULT`, or a named preset JSON in `GalCraft/instrument`. |
| `spatial_resolution` | Angular pixel size `[x, y]` in arcsec; overridden by a named instrument preset. |
| `spatial_nbin` | Number of spaxels `[nx, ny]` for each cube, except in `DEFAULT` mode. |
| `spatial_percentile` | In `DEFAULT` mode, use the 0.05–99.95 percentiles instead of the full data extent. |
| `bootstrap_table` | Enable the implemented bootstrap treatment of the particle table. |
| `use_dist` | Use per-particle `r` rather than one adopted galaxy distance for flux scaling. |
| `use_losvd` | Include stellar line-of-sight velocities in generated spectra. |
| `use_extinc` | Apply attenuation from input `exbv`; requires that field. |
| `extinc_factor` | Scale the extinction applied to the spectra. |
| `invert_b` | Spatial-coordinate inversion flag handled during binning. |
| `sn` | Target maximum signal-to-noise parameter used by the synthetic-noise calculation. |
| `add_noise` | Add simulated Gaussian flux noise and write an `ERROR` extension. |

**Instrument modes.** `DIY` uses spatial settings from the setup JSON; a named instrument loads `<instrument>.json` and overrides matching keys. `DEFAULT` spans the catalogue's sky-coordinate extent and does not use the cube-centre list. Other modes expect a CSV-style file named `setup_<config>_list` containing two coordinates per cube centre.

## Execution and mass scaling: `other_params`

| Key | Meaning |
| --- | --- |
| `mode` | `normal`, `debug`, or `continue`. |
| `model_name` | Particle catalogue filename, relative to `modelDir` (EBF format). |
| `ncpu` | Multiprocessing worker count or `null` for available CPU count. |
| `nprocess_ploop` | Size of the spatial-pixel batch handed to workers. |
| `nparticles` | Optional particle count for mass normalisation; `null` uses the input count. |
| `gal_mass` | Optional total galaxy stellar mass for flux scaling; if `null`, no global scaling is applied. |
| `log_mode` | Log file mode (`w` or `a`); forced to `a` for continuation. |
| `write_table` | Save intermediate particle tables; required if future continuation is intended. |
| `plot_statistics` | Toggle implemented statistical diagnostic plotting. |

The mass-calibration scale in the implementation is `gal_mass / nparticles`. In the standard binning branch, the code sets each particle's `mass` to one before this global scaling. Ensure that this convention is appropriate for the model being simulated.

:::{warning}
This table describes the present implementation, not an independently validated physical forward model for every input catalogue. Check parameter conventions, template flux normalization, and distance/mass assumptions before interpreting absolute fluxes.
:::
