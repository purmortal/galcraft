# Scientific method and interpretation

GalCraft is an **observation simulator**, not a stellar-population inference tool. It maps stellar particles onto an apparent sky-plane grid and computes an integrated optical spectrum per spatial bin using an SSP library.

## Stellar-population assignment

The standard particle-catalogue branch converts iron abundance and alpha enhancement to a total metallicity estimate via the implemented expression

\[ [M/H] = [Fe/H] + \log_{10}\left(0.684 \times 10^{[\alpha/Fe]} + 0.306\right). \]

Ages, `[M/H]` and `[alpha/Fe]` are clipped to the template grid. The user selects the SSP interpolation method in the configuration.

## Geometry and stellar kinematics

`Cot` rotates and translates the galaxy from its native reference frame to an observer placement. Spatial binning converts angular pixel sizes from arcseconds to degrees. When `use_losvd` is enabled, per-particle line-of-sight velocities affect the integrated pixel spectra. The implementation uses `doppler_shift` to map velocity to spectral shifts.

## Spectral resolution

SSP spectra are oversampled and convolved with a broadening kernel to match the configured target instrumental resolution. Broadening is computed from the quadrature difference between instrumental and template FWHM, with the Gaussian dispersion in wavelength pixels.

:::{important}
GalCraft broadens a higher-resolution template to a lower-resolution target; convolution cannot physically recover spectral detail finer than that present in a template. Wavelength regions requiring negative broadening variances are discarded or rejected by the loader.
:::

## Dust extinction and noise

If requested, the simulated dust attenuation uses the Calzetti et al. (2000) prescription, multiplying spectra by a wavelength-dependent attenuation factor computed from `exbv` and the extinction scaling option. If `add_noise` is enabled, the writer uses a simplified square-root noise prescription and produces an `ERROR` extension. This is not a complete model of detector/readout/sky-subtraction systematics.

## Further reading

- [Wang et al. (2024), *Validating full-spectrum fitting with a synthetic integral-field spectroscopic observation of the Milky Way*](https://doi.org/10.1093/mnras/stae2148)
- [GalCraft source code](https://github.com/purmortal/galcraft)

For quantitatively calibrated forward modelling, cite and consult the GalCraft methods paper in addition to the source code and configuration of the run.
