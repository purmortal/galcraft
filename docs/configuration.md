# Configuration

GalCraft uses **two files** for each run: a directory map and a JSON setup file. The directory map tells the program where to find data. The JSON setup contains four dictionaries: `oparams`, `ssp_params`, `cube_params` and `other_params`.

## How configuration files are selected

```bash
GalCraft --config example --default-dir /path/to/defaultDir
```

This command loads `<configDir>/setup_example.json` and creates outputs under `<outputDir>/example/`. If the instrument is not `DEFAULT`, it also reads a comma-separated cube-centre file `<configDir>/setup_example_list`.

### Directory map (`defaultDir`)

```ini
configDir = /absolute/path/to/configFiles/
modelDir = /absolute/path/to/models/
templateDir = /absolute/path/to/templates/
outputDir = /absolute/path/to/output/
```

| Name | Used for |
| --- | --- |
| `configDir` | `setup_NAME.json` and, for non-`DEFAULT` runs, `setup_NAME_list`. |
| `modelDir` | EBF stellar-particle catalogue named by `other_params.model_name`. |
| `templateDir` | SSP library files matching `ssp_params.library`. |
| `outputDir` | New `NAME/` directory with cubes, diagnostics and logs. |

**Use absolute paths to existing directories with trailing `/`**. Parts of the implementation concatenate these directory strings rather than normalising them automatically.

## Observer geometry: `oparams`

These parameters place/rotate a galaxy model into the observer frame **only if** the input catalogue lacks a `vr` field. If `vr` is already present, GalCraft assumes observer-frame coordinates and sets the in-memory observer values to `null`.

| Setting | Type / units | Explanation |
| --- | --- | --- |
| `distance` | number, kpc | Adopted observer-to-galaxy distance; also used for uniform distance scaling when `use_dist` is `false`. |
| `l` | number, deg | Galactic longitude of the model centre. |
| `b` | number, deg | Galactic latitude of the model centre. |
| `theta_zx` | number, deg | First rotation angle passed to the observer coordinate transformation. |
| `theta_yz` | number, deg | Second rotation angle passed to the observer coordinate transformation. |

These are **model-orientation** parameters, not sky WCS settings for the output cube; the resulting binned coordinates determine the FITS spatial header.

## SSP spectra: `ssp_params`

| Setting | Type / units | Explanation |
| --- | --- | --- |
| `model` | string | Reader identifier; the code imports `GalCraft.ssp_utils.<model>_util`. Available readers are listed below. |
| `library` | string / glob | Path pattern relative to `templateDir` used to select spectral template files. Its conventions depend on the chosen reader. |
| `factor` | integer | Oversampling factor for spectral processing/broadening; higher factors increase memory and computation. |
| `FWHM_gal` | float (Å) or filename | Requested output/instrumental line-spread FWHM, either constant or wavelength-dependent file. |
| `FWHM_tem` | float (Å) or reader-supported file | Native template spectral FWHM used in relative broadening calculations. |
| `dlam` | float, Å, or `null` | Requested linear wavelength sampling increment, or preserve library grid when `null`. |
| `age_range` | `[min,max]` or `null` | Optional SSP age range (Gyr); `null` means full library availability. |
| `metal_range` | `[min,max]` or `null` | Optional SSP metallicity selection (dex), otherwise use available grid. |
| `alpha_range` | `[min,max]` or `null` | Optional SSP alpha-enhancement selection (dex), otherwise use available grid. |
| `wave_range` | `[min,max]` or `null` | Optional final FITS wavelength limits in Å; applied by the cube writer. |
| `spec_interpolator` | string | `nearest`, `linear` or `interp_on_alpha`; latter uses a specialised branch with a known implementation issue. |

**Resolution and grid limits.** The code cannot restore lost spectral resolution: requests for output broadening finer than the intrinsic templates can fail or exclude invalid wavelength regions. Outside-grid stellar population values are clipped to available SSP axes rather than extrapolated.

### SSP reader and input data

| `model` value | Reader implementation |
| --- | --- |
| `miles` | `ssp_utils/miles_util.py` |
| `conroyinmiles` | `ssp_utils/conroyinmiles_util.py` |
| `pegasehr` | `ssp_utils/pegasehr_util.py` |
| `pegasehr_interp` | `ssp_utils/pegasehr_interp_util.py` |
| `xshooter` | `ssp_utils/xshooter_util.py` |
| `alphaMC` | `ssp_utils/alphaMC_util.py` |

SSP libraries are not included in the source ZIP. Each reader expects specific spectral arrays, filename conventions, wavelength sampling and population grid definitions. Ages in SSP reader grids are in Gyr, while interpolation works with `log10(age in years)`.

## Spatial cubes and physical effects: `cube_params`

| Setting | Type / units | Explanation |
| --- | --- | --- |
| `x_coord` | string | Name of catalogue coordinate for x-axis, commonly `l` or `ra`. |
| `y_coord` | string | Name of catalogue coordinate for y-axis, commonly `b` or `dec`. |
| `instrument` | string | `DIY`, `DEFAULT`, or preset identifier such as `MUSE-WFM`. |
| `spatial_resolution` | `[x,y]`, arcsec | Angular spaxel dimensions. A named instrument preset overrides this field. |
| `spatial_nbin` | `[nx,ny]`, integers | Numbers of spatial bins for each selected cube centre, except `DEFAULT` mode. Presets may override this field. |
| `spatial_percentile` | bool | For `DEFAULT`, optionally bound the x/y footprint at the 0.05th and 99.95th percentiles instead of full min/max. |
| `bootstrap_table` | bool | Use the implementation's bootstrap resampling of the particle catalogue during spatial binning. |
| `use_dist` | bool | Scale each particle contribution using its own distance (`r`) when `true`; otherwise use `oparams.distance`. |
| `use_losvd` | bool | Apply each particle's line-of-sight velocity `vr` to its contributing spectrum. |
| `use_extinc` | bool | Apply extinction from the `exbv` column; the run asserts that field exists when enabled. |
| `extinc_factor` | number | Extinction scaling divisor used in the pixel generator (`exbv / extinc_factor`); avoid zero. |
| `invert_b` | bool | Negate `b` during binning; relevant to the selected coordinate orientation. |
| `sn` | number | Maximum S/N-related parameter for the simplified noise model; not used to add noise unless `add_noise` is `true`. |
| `add_noise` | bool | Add generated Gaussian noise to `DATA` and write an `ERROR` image containing its standard deviation. |

### Instrument modes

| Value | How the footprint is made | Required extra file |
| --- | --- | --- |
| `DIY` | Uses `spatial_resolution` and `spatial_nbin` from your JSON. | `setup_NAME_list` with cube centres. |
| A named preset, e.g. `MUSE-WFM` | Loads `GalCraft/instrument/<preset>.json` and overrides matching spatial keys. | `setup_NAME_list`. |
| `DEFAULT` | Builds a single covering grid from coordinate extent or specified percentiles; `spatial_nbin` is not used. | None. |

The centre-list file must contain **two comma-separated sky-coordinate values per line**, in degrees, matching `x_coord` and `y_coord`, for example:

```text
15.20,-27.30
15.21,-27.30
```

A named instrument preset specifies *spatial* parameters only; the spectral LSF is separately chosen through `ssp_params.FWHM_gal`. The included `MUSE-WFM.json` currently declares `[0.2, 0.2]` arcsec pixels and `[315, 315]` spatial bins.

## Run mode and mass scaling: `other_params`

| Setting | Type / units | Explanation |
| --- | --- | --- |
| `mode` | string | `normal`, `debug`, or `continue`. `normal` is recommended for scientific production. |
| `model_name` | string | EBF catalogue filename relative to `modelDir`. |
| `ncpu` | positive integer or `null` | Worker count; `null` selects all CPUs reported by multiprocessing. |
| `nprocess_ploop` | positive integer | Number of nonempty spaxels submitted per multiprocessing batch. |
| `nparticles` | integer or `null` | Particle count used as denominator for global mass calibration; `null` normally uses the catalogue count. |
| `gal_mass` | number or `null` | Total galaxy stellar mass used for the calibration multiplier `gal_mass / nparticles`; `null` disables global multiplication. |
| `log_mode` | `w` or `a` | Overwrite/append log file. `continue` forces append mode. |
| `write_table` | bool | Save intermediate `particle_table_<i>.fits` files, required for future `continue` runs. |
| `plot_statistics` | bool | Enable additional input particle-population diagnostic plots. |

### Normal, debug and continue

- **`normal`:** complete simulation, including initial spatial binning.
- **`debug`:** special sampled-particle path; should not be treated as equivalent to the full galaxy result.
- **`continue`:** skips initial binning, reloads intermediate particle tables/count arrays and bypasses existing `data_cube_<i>.fits`. It requires an earlier run with `write_table: true` and is not supported for `instrument: DEFAULT`.

### Mass and distance interpretation

In the normal spatial-binning branch, GalCraft sets each particle's mass weight to one. If `gal_mass` is supplied, it multiplies the assembled cube by `gal_mass / nparticles`. `use_dist` controls whether the contributing stellar spectra use per-particle or one adopted distance. Confirm that both conventions suit your particle catalogue before interpreting physical fluxes.

## Input EBF catalogue

`MainProcess` loads the particle catalogue with `ebf.read(..., '/')`. It is **not** a general-purpose arbitrary-FITS catalogue interface. Depending on whether it is already observer-transformed, required fields include:

- **Observer-frame:** `l`, `b`, `ra`, `dec`, `vr`, `r`, and the stellar population fields used below.
- **Untransformed galaxy coordinates:** `Cot.observe` expects fields including `lgc`, `pzgc`, `rcgc`, with velocity fields such as `vlgc`, `vzgc`, `vrcgc` when available.
- **Standard stellar population fields:** `m_ini`, `fe_h`, `alpha_fe`, `age`. GalCraft derives `[M/H]`, clips population parameters to the SSP limits, and calculates `cube_logage`.
- **Prepared stellar population branch:** fields including `cube_m_h`, `cube_alpha_fe`, `cube_age` may already be supplied.
- **Optional dust:** `exbv` is required if `use_extinc` is `true`.

Do not assume that columns can be renamed without modifying the reader and binning code. Validate physical units and template flux normalisation against your source simulations.

## Configuration checks before running

1. All four directories exist and their paths have trailing slashes.
2. SSP `model` matches the template library files selected by `library`.
3. Requested `FWHM_gal`, `dlam` and `wave_range` are appropriate for the SSP wavelength/resolution grid.
4. The catalogue contains the coordinate, population, distance and optional extinction columns required by enabled features.
5. The centre-list file exists when using `DIY` or a named instrument.
6. `gal_mass`, `nparticles`, `distance` and `use_dist` agree with the desired physical normalisation.
7. For `continue`, the initial simulation used `write_table: true`.

Start from the [official test-kit configuration](https://github.com/purmortal/galcraft/tree/test_kit/tests/configFiles) instead of inventing new required keys from scratch.
