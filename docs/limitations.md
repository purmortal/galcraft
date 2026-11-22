# Known limitations and validation checklist

The following observations come from inspection of the uploaded code, **not** from an end-to-end science validation. They are documented here so users do not confuse current behavior with general API guarantees.

## Specific code issues found

- `Cot.gal2equ(l, b)` refers to undefined `mua` and `mud` in its two-argument branch. Use caution with this helper; the velocity-aware branch is different.
- `Cot.xyz2lbr(px, py, pz)` refers to an undefined `r` in its three-argument return branch (it computes `r1`).
- `SpecGeneratorPix.setup_interpolator()` returns a local variable `interpolator` that is not assigned in the `interp_on_alpha` branch, even though it assigns `self.interpolator` there. Treat that special branch as unverified and prefer the tested interpolation configuration.
- `MainProcess.main(args=None)` does not pass its `args` argument into `parser.parse_args()`; programmatic callers should not assume they can override `sys.argv` through that parameter.
- The workflow tests in `.github/workflows/test_build.yml` use external example data from the `test_kit` branch. The present documentation build validates source documentation, not scientific result agreement.

## User checks before publication-quality simulations

1. Confirm the input EBF field names and galaxy coordinate system.
2. Verify SSP-template age, abundance, wavelength, native FWHM and flux conventions.
3. Check mass scaling (`gal_mass / nparticles`) and distance choices.
4. Confirm whether `use_losvd`, `use_extinc`, `use_dist` and `add_noise` match your experiment.
5. Compare output FITS `DATA` and `WAVE` axes and generated plots against expected instrument properties.
6. Set a controlled random seed externally if exact reproducibility of any randomised steps matters; the current CLI does not expose a seed option.

These are **documentation findings**. The delivered archive intentionally leaves executable scientific logic unchanged; fixes should be considered and tested separately.
