# Input data and SSP libraries

## Particle catalogues

The runner uses `ebf.read(<modelDir>/<model_name>, '/')` to obtain the model data. GalCraft is designed around an E-Galaxia-style **EBF stellar-particle catalogue**, rather than accepting an arbitrary FITS table at the command line.

There are two branches of the input treatment:

- **Galaxy-internal coordinates:** if the catalogue lacks `vr`, `Cot.observe` uses the observer parameters to transform galaxy-centred coordinates to the observer frame. The coordinate utility expects fields including `lgc`, `pzgc` and `rcgc`, and optionally `vlgc`, `vzgc`, `vrcgc` for velocities.
- **Already transformed catalogue:** if `vr` is present, GalCraft assumes the catalogue is already in observer coordinates and does not perform a rotation.

The spatial binner expects observable coordinate fields including `l`, `b`, `ra`, `dec`, `vr` and `r`, plus the stellar-population fields needed by the chosen branch. In the standard branch these include `m_ini`, `fe_h`, `alpha_fe` and `age`; the code derives `m_h` and clips ages and abundances onto the available SSP grid. A preprocessed branch instead uses the existing `cube_m_h`, `cube_alpha_fe`, and `cube_age` fields.

If extinction is enabled, the table must include `exbv`. Do not rename fields without updating the code that consumes them.

## SSP libraries

The SSP selection is controlled by `ssp_params.model` and `ssp_params.library` (a glob relative to `templateDir`). Available readers in the uploaded source include:

- `miles` (MILES / E-MILES-style filename handling)
- `conroyinmiles`
- `pegasehr` and `pegasehr_interp`
- `xshooter`
- `alphaMC`

The library data are **not** shipped with the main archive. Obtain the appropriate files and ensure their filename and axis conventions match the selected reader. In all cases the output of the library reader is expected to expose `templates`, `lam_temp`, population grids and `FWHM_tem` for the main loader.

SSP ages are represented in Gyr in the template-reader grids; the spectral interpolation uses `log10(age [yr])`. Metallicities and alpha enhancements are represented in dex. Input values outside the library's ranges are clipped by the binning stage; this can accumulate particles in boundary templates.

## Instrument files

Preset spatial parameters are in `GalCraft/instrument/*.json`. The repository also contains wavelength-dependent LSF text files (e.g. `lsf_MUSE-WFM-GIST`). LSF files are read with `numpy.genfromtxt`, with wavelengths in the first column and FWHM values in the second.
