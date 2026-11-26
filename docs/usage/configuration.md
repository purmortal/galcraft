# Configuration

GalCraft uses **three configuration files** for each run: a directory mapping file (`defaultDir`), a JSON setup file (`setup_NAME.json`), and a list of data cube pointings (`setup_NAME_list`).

The directory mapping file specifies where to find the input data and save the outputs. The JSON setup file defines the simulation parameters through four dictionaries: `oparams`, `ssp_params`, `cube_params`, and `other_params`. The pointing list specifies the centre coordinates of the data cubes to be generated.

Example configuration files are available in the GalCraft test kit:

- **Directory mapping file:** [`defaultDir`](https://github.com/purmortal/galcraft/blob/test_kit/tests/configFiles/defaultDir)
- **JSON setup file:** [`setup_test_kit.json`](https://github.com/purmortal/galcraft/blob/test_kit/tests/configFiles/setup_test_kit.json)
- **Data cube pointing list:** [`setup_test_kit_list`](https://github.com/purmortal/galcraft/blob/test_kit/tests/configFiles/setup_test_kit_list)


## How configuration files are selected

```bash
GalCraft --config NAME --default-dir /path/to/defaultDir
```

This command loads `<configDir>/setup_NAME.json` and creates outputs under `<outputDir>/NAME/`.

### Directory map (`defaultDir`)

```ini
configDir = /absolute/path/to/configFiles/
modelDir = /absolute/path/to/models/
templateDir = /absolute/path/to/templates/
outputDir = /absolute/path/to/output/
```

| Name | Used for                                                                                                     |
| --- |--------------------------------------------------------------------------------------------------------------|
| `configDir` | Includes `setup_NAME.json` and `setup_NAME_list`.                                                            |
| `modelDir` | Includes stellar particle catalogs named by `other_params.model_name`.                                       |
| `templateDir` | SSP library files matching `ssp_params.library`.                                                             |
| `outputDir` | Directory of output files, to be saved under `<outputDir>/NAME/`. |

## Observer geometry: `oparams`

These parameters place/rotate a galaxy model into the observer frame.

| Setting | Type / units | Explanation |
| --- | --- | --- |
| `distance` | number, kpc | Adopted observer-to-galaxy distance; also used for uniform distance scaling when `use_dist` is `false`. |
| `l` / `ra` | number, deg | Galactic longitude (`l`) or right ascension (`ra`) of the model centre. |
| `b` / `dec` | number, deg | Galactic latitude (`b`) or declination (`dec`) of the model centre. |
| `theta_zx` | number, deg | Rotates the galaxy from the z-axis towards the x-axis (positive) or vice versa (negative). |
| `theta_yz` | number, deg | Rotates the galaxy from the y-axis towards the z-axis (positive) or vice versa (negative). |


## SSP spectra: `ssp_params`

These parameters control the selection, interpolation, and spectral resolution of the SSP templates used to generate the mock data cubes.

| Setting | Type / units | Description                                                                                                                                                                                                                                                                                                                                                                                |
| --- | --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model` | string | SSP library identifier: `miles`, `pegasehr`, `alphaMC`, `xshooter`.                                                                                                                                                                                                                                                                                                                        |
| `library` | string | Path or filename pattern of the SSP templates, relative to `templateDir`. For example, `miles/MILES_BASTI_KU_Ep*/*.fits`.                                                                                                                                                                                                                                                                  |
| `factor` | integer | Oversampling factor for the SSP wavelength grid. For example, `10` increases the spectral sampling by a factor of 10.                                                                                                                                                                                                                                                                      |
| `FWHM_gal` | number, Å, or string | Target spectral resolution (FWHM) of the final data cube, specified as a constant (e.g. 2.51 for MILES) or a wavelength-dependent LSF. Example LSF files are available in the [instrument folder](https://github.com/purmortal/galcraft/tree/main/GalCraft/instrument). For example, `lsf_MUSE-WFM` means MUSE Wide Field Mode LSF. Users can also add their own LSF files to this folder. |
| `FWHM_tem` | number, Å, or string | Intrinsic spectral resolution (FWHM) of the SSP templates. MILES has a constant FWHM of 2.51 Å, while PEGASE-HR, α-MC, and X-shooter use `lsf_PEGASEHR`, `lsf_alphaMC`, and `lsf_xshooter`, respectively.                                                                                                                                                                                  |
| `dlam` | number, Å, or `null` | Wavelength sampling interval of the final data cube. If `null`, the original SSP wavelength grid is used.                                                                                                                                                                                                                                                                                  |
| `age_range` | `[min, max]`, Gyr, or `null` | Age range of the selected SSP templates. For example, `[0.1, 14]` in Gyr. If `null`, the full available range is used.                                                                                                                                                                                                                                                                     |
| `metal_range` | `[min, max]`, dex, or `null` | Metallicity range of the selected SSP templates. If `null`, the full available range is used.                                                                                                                                                                                                                                                                                              |
| `alpha_range` | `[min, max]`, dex, or `null` | α-element enhancement range of the selected SSP templates. If `null`, the full available range is used.                                                                                                                                                                                                                                                                                    |
| `wave_range` | `[min, max]`, Å, or `null` | Wavelength range of the final data cube. If `null`, the full available range is used.                                                                                                                                                                                                                                                                                                      |
| `spec_interpolator` | string | Interpolation method for assigning SSP spectra to particles. Options include `linear`, `nearest`, `slinear`, `cubic`, and `quintic`. See the [SciPy RegularGridInterpolator documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html) for details.                                                                               |

**Resolution and grid limits.** The code cannot restore lost spectral resolution: requests for output broadening finer than the intrinsic templates can fail or exclude invalid wavelength regions. Outside-grid stellar population values are clipped to available SSP axes rather than extrapolated.

Please ensure that you download the required SSP templates and save them in your chosen directory before running GalCraft. Note that some readers require specially formatted template files; contact the developer if you need assistance obtaining compatible files.

## Data cube: `cube_params`

These parameters define the spatial dimensions, coordinate system, and physical effects included in the mock IFS data cubes.

| Setting | Type / units | Description                                                                                                                                                                                                      |
| --- | --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x_coord` | string | Coordinate used for the x-axis of the data cube: `l` or `ra`.                                                                                                                                                    |
| `y_coord` | string | Coordinate used for the y-axis of the data cube: `b` or `dec`.                                                                                                                                                   |
| `instrument` | string | IFS instrument configuration, such as `MUSE-WFM`, or `DIY` for user-defined settings. `DEFAULT` is also available for an automatically determined spatial grid.                                                  |
| `spatial_resolution` | `[x, y]`, arcsec | Spatial pixel size along the x- and y-axes. Used when `instrument` is `DIY`.                                                                                                                                     |
| `spatial_nbin` | `[nx, ny]`, integers | Number of spatial pixels along the x- and y-axes. Used when `instrument` is `DIY`.                                                                                                                               |
| `spatial_percentile` | boolean | Controls percentile-based spatial limits in `DEFAULT` mode. Normally set to `true`.                                                                                                                              |
| `bootstrap_table` | boolean | Enables particle bootstrap resampling for estimating sampling uncertainties. Normally set to `false`. See Section 2.4 of [Wang et al. (2024)](https://doi.org/10.1093/mnras/stae2148) for detailed descriptions. |
| `use_dist` | boolean | If `true`, accounts for the individual distance of each stellar particle when calculating its flux contribution.                                                                                                 |
| `use_losvd` | boolean | If `true`, applies the Doppler shift associated with each particle's line-of-sight velocity. If `false`, no kinematic information is included in the final data cube. |
| `use_extinc` | boolean | If `true`, applies dust extinction to individual stellar particles using the Calzetti et al. (2000) attenuation law (Equations 2–5). The line-of-sight integrated E(B−V) must be provided in the input stellar particle catalogue (`exbv`). |
| `extinc_factor` | number | Scaling factor applied to the stellar particle E(B−V) values in the extinction calculation.                                                                                                                      |
| `invert_b` | boolean | Reverses the sign of the Galactic latitude coordinate, if required by the galaxy orientation. Normally set to `false`. |
| `sn` | number | Target signal-to-noise ratio of the brightest spaxel, used to determine the noise level across the data cube.                                                                                                    |
| `add_noise` | boolean | If `true`, adds simulated noise and produces a flux-error array. If `false`, the data cube contains only the simulated flux.                                                                                     |





### Instrument configurations

GalCraft provides three options for defining the spatial coverage of a mock observation:

- **`DIY`:** Uses the user-defined `spatial_resolution` and `spatial_nbin` parameters.
- **Instrument preset:** Uses the spatial configuration of a predefined instrument, such as `MUSE-WFM`, overriding the corresponding user-defined spatial parameters.
- **`DEFAULT`:** Automatically determines a spatial grid covering the stellar particle distribution.

For example, the `MUSE-WFM` preset uses a spatial pixel scale of 0.2 × 0.2 arcsec² and a grid of 315 × 315 spaxels.

Additional instrument configurations, including MaNGA, SAMI, and Hector, are planned for future releases.

```{important}
The `instrument` parameter controls the spatial configuration of the data cube, while the spectral resolution is specified separately through `ssp_params.FWHM_gal`.
```
### Noise and physical effects

GalCraft allows users to enable or disable the following effects independently:

- **Distance (`use_dist`):** Accounts for differences in the distances of individual stellar particles.
- **Line-of-sight velocity (`use_losvd`):** Doppler-shifts each particle's spectrum according to its line-of-sight velocity.
- **Extinction (`use_extinc`):** Applies dust extinction based on the particle's E(B−V) value.
- **Noise (`add_noise`):** Adds Gaussian noise to the final data cube, with the brightest spaxel having the reference S/N specified by `sn`.

These settings allow users to generate data cubes with different levels of observational complexity.


## Execution and processing: `other_params`

These parameters control the GalCraft execution, multiprocessing, input catalogue, mass scaling, and additional outputs.

| Setting | Type / units | Description                                                                                                               |
| --- | --- |---------------------------------------------------------------------------------------------------------------------------|
| `mode` | string | Execution mode: `normal`, `debug`, or `continue`.                                                                         |
| `model_name` | string | Filename of the input stellar particle catalogue, relative to `modelDir`.                                                 |
| `ncpu` | integer or `null` | Number of CPU cores used for parallel processing. If `null`, all available CPU cores are selected.                        |
| `nprocess_ploop` | integer | Number of spaxels processed in each multiprocessing batch.                                                                |
| `nparticles` | integer or `null` | Total number of stellar particles used for mass normalisation. If `null`, the particle count is determined automatically. |
| `gal_mass` | number, M☉, or `null` | Total stellar mass of the input galaxy, used to normalise the particle contributions.                                     |
| `log_mode` | `w` or `a` | Log-file writing mode: `w` overwrites the existing log, while `a` appends to it.                                          |
| `write_table` | boolean | If `true`, saves the stellar particle table for each generated data cube as a FITS file.                                  |
| `plot_statistics` | boolean | If `true`, generates diagnostic plots of the input stellar particle distributions.                                        |


### Execution modes

GalCraft supports three execution modes:

- **`normal`:** Runs the complete task using the full input stellar particle catalogue. Recommended for scientific applications.
- **`debug`:** Runs a reduced sample of 50,000 stellar particles to test the configuration before performing a full run.
- **`continue`:** Resumes a previously interrupted run using saved files and setups in `<outputDir>/NAME`, skipping data cubes that have already been generated.

```{important}
To use `continue` mode, the previous run must have `write_table` set to `true`, so that the intermediate stellar particle tables are available.
```

### Parallel processing

GalCraft uses multiprocessing to generate integrated spectra across spatial bins.

The `ncpu` parameter specifies the number of CPU cores, while `nprocess_ploop` determines the number of spaxels processed per batch.

For example, if a data cube contains 10,000 spaxels and `nprocess_ploop` is set to `50`, GalCraft processes the spaxels in 200 batches.


### Stellar mass normalisation

GalCraft uses `gal_mass` and `nparticles` to define the mass normalisation of the input galaxy.

When both are specified, the global mass-scaling factor is calculated as:

$$
f_{\mathrm{mass}} = \frac{\mathrm{gal\_mass}}{\mathrm{nparticles}}.
$$

If `nparticles` is `null`, GalCraft determines the particle count automatically. If `gal_mass` is `null`, the global mass-scaling factor is not applied.


## Data cube pointings

When using `DIY` or a predefined instrument configuration, GalCraft requires a `setup_NAME_list` file specifying the central coordinates of each data cube.

Each line should contain two comma-separated coordinates in degrees, corresponding to `x_coord` and `y_coord`.

For example:

```text
15.20,-27.30
15.21,-27.30
```

In this example, GalCraft generates two data cubes centred at the specified coordinates.

When `instrument` is set to `DEFAULT`, the spatial grid is determined automatically, and a pointing list is not required.
