# Simulation workflow

The command-line execution follows these steps:

1. **Load configuration.** `MainProcess.run_GalCraft` reads `setup_<name>.json`, resolves the four directories, and starts a log.
2. **Load instrument spatial settings.** A named preset, `DIY`, or `DEFAULT` determines the pixel grid (see [configuration](configuration.md)).
3. **Prepare SSP templates.** `SSPLoader.model` loads the selected library, derives the population axes, checks target versus template spectral resolution, and oversamples spectra for broadening.
4. **Read particles.** The input EBF file is read using `ebf.read`. If the catalogue does not have a `vr` field, the code applies the configured observer transformation with `Cot.observe`.
5. **Spatially bin particles.** `Binner.spatial_binner` assigns particles to cube centres/spaxels, clips their stellar-population parameters to the template grid, writes particle-count arrays, and optionally saves tables.
6. **Synthesize spaxel spectra.** `SpecGenerator.Spec_Generator` dispatches nonempty pixels across multiple processes. `SpecGeneratorPix` computes integrated spectra and SSP population weights.
7. **Write FITS cubes.** `CubeMaker.write_cube` optionally trims wavelengths, adds noise, and writes the `DATA`, optional `ERROR`, and `WAVE` extensions.
8. **Save provenance.** Outputs include `setup.json`, `outputs.log`, NumPy population-weight/count arrays, and diagnostic figures.

## Modes

- `normal` — standard execution. Run this first.
- `debug` — randomly samples a fixed number of particles in the implementation; not suitable for a production result or calibrated mock observation without checking its assumptions.
- `continue` — reloads previously written particle tables and count arrays, skips existing FITS cube files, and processes remaining cubes. It requires `write_table: true` on the earlier run and is not supported with `instrument: DEFAULT`.

## Parallelism

`other_params.ncpu` selects the multiprocessing worker count; `null` uses `multiprocessing.cpu_count()`. `other_params.nprocess_ploop` controls batches of spatial pixels submitted to workers. Very large SSP grids and many workers can increase memory usage substantially.
