# Workflow

GalCraft runs a sequence of modules from a command-line entry point. The overview below follows the execution order in `GalCraft/MainProcess.py` and distinguishes the orchestration layer from the modules that perform scientific operations.

## Pipeline overview

```text
GalCraft (command line)
  │
  ▼
MainProcess.main → MainProcess.run_GalCraft
  ├── utils.initialize_Dirs / setupLogfile  → paths, JSON, logging, CPUs
  ├── instrument/*.json                    → optional spatial preset
  ├── SSPLoader.model / ssp_utils/*         → SSP library + LSF + spectral grid
  ├── ebf.read / Cot.observe               → input particles + observer geometry
  ├── Binner.spatial_binner                 → particle tables, spaxels, counts
  │     └── Binner.spatial_binner_continue  → reuse tables in continue mode
  ├── SpecGenerator.Spec_Generator         → dispatch spaxels to workers
  │     └── SpecGeneratorPix               → SSP weights and spectra per spaxel
  │           └── utils                    → Doppler shift, attenuation, maps
  ├── CubeMaker.write_cube                 → FITS DATA / ERROR / WAVE
  └── setup.json / outputs.log             → provenance and diagnostics
```

## Processing stages

| Step | Module | Description and main products |
| --- | --- | --- |
| 1. Initialise | `MainProcess`, `modules.utils` | Parse CLI options, resolve directories, load `setup_NAME.json`, prepare output directory and logging, select CPU count. |
| 2. Choose instrument | `MainProcess`, `instrument/*.json` | Use `DIY`, `DEFAULT`, or an instrument preset to determine spatial pixel dimensions. |
| 3. Prepare SSP templates | `modules.SSPLoader`, `ssp_utils` | Load SSP spectra, inspect age/metallicity/alpha grids, reconcile spectral resolution, prepare an oversampled wavelength grid. |
| 4. Load and position particles | `ebf`, `modules.Cot` | Read the EBF catalogue; if `vr` is absent, transform from galaxy-centred coordinates using the observer geometry. |
| 5. Bin spatially | `modules.Binner` | Select particles within each cube footprint, map them to spaxels, clip template parameters to available SSP ranges, optionally write particle tables. |
| 6. Generate spectra | `modules.SpecGenerator`, `modules.SpecGeneratorPix` | Schedule non-empty spaxels across processes, interpolate SSPs and combine particle contributions; optionally apply line-of-sight shifts and extinction. |
| 7. Write cube | `modules.CubeMaker` | Apply optional final wavelength trimming and simplified Gaussian noise, then serialize FITS cube extensions and WCS metadata. |
| 8. Save diagnostics | `modules.utils`, `MainProcess` | Record NumPy arrays, plots, the effective `setup.json` and `outputs.log`. |

## Modules included

### Entry point and orchestration

- **`GalCraft.MainProcess`** — `main()` parses `--config` and `--default-dir`; `run_GalCraft()` orchestrates a complete simulation. It is the installed CLI entry point.
- **`GalCraft._version`** — exposes the package version written into provenance metadata.

### Core scientific modules

- **`GalCraft.modules.Cot`** — coordinate and velocity transformations, including `observe()` for placing a simulated galaxy relative to an observer.
- **`GalCraft.modules.SSPLoader`** — `model` class: dynamically selects SSP readers, manages grids, matches spectral resolutions and prepares templates for pixel synthesis.
- **`GalCraft.modules.Binner`** — `spatial_binner()` and `spatial_binner_continue()` assign particles to spatial bins or reload intermediate tables for continuation.
- **`GalCraft.modules.SpecGenerator`** — `Spec_Generator` schedules the non-empty spaxels for multiprocessing and assembles their spectra and SSP weight arrays.
- **`GalCraft.modules.SpecGeneratorPix`** — generates each spaxel spectrum using SSP interpolation, mass/distance scaling, velocities and optional dust attenuation.
- **`GalCraft.modules.CubeMaker`** — `write_cube()` adds optional noise and produces the final FITS file and headers.
- **`GalCraft.modules.utils`** — shared directory parsing, logging, spectral shifts, attenuation and diagnostic plotting utilities.
- **`GalCraft.modules.constant`** — common physical constants imported by the supporting calculations.

### SSP-reader modules

The `ssp_params.model` field selects the corresponding `<model>_util.py` implementation under `GalCraft/ssp_utils/`:

| Reader name | Module | Purpose |
| --- | --- | --- |
| `miles` | `miles_util` | MILES-style SSP templates. |
| `conroyinmiles` | `conroyinmiles_util` | Conroy-related templates in the supported MILES-style format. |
| `pegasehr` | `pegasehr_util` | PEGASE-HR spectra. |
| `pegasehr_interp` | `pegasehr_interp_util` | Interpolated PEGASE-HR-related templates. |
| `xshooter` | `xshooter_util` | X-shooter-based SSP library files. |
| `alphaMC` | `alphaMC_util` | Alpha-enhanced model spectra supported by this reader. |

The SSP reader and its input filename conventions must match the actual template library; readers are not interchangeable with arbitrary spectral files. Their functions are listed on the [Python API](api.md) page.

## Spatial and spectral logic

A cube has spatial dimensions defined by the chosen centre list and `spatial_nbin`, except in `DEFAULT` mode, which derives its footprint from the catalogue. Each nonempty spatial bin receives a summed spectrum. The loader prepares SSP grids that the per-pixel generator uses for stellar ages, metallicities and alpha enhancements. Particle velocities are included only when `use_losvd` is enabled; extinction is used only when `use_extinc` is enabled and the catalogue supplies `exbv`.

The spectral cube starts in memory as `(x, y, wavelength)` and is written to FITS in `(wavelength, y, x)` order when read by Astropy.

## Run modes and parallel processing

| Mode | Behavior | Caveat |
| --- | --- | --- |
| `normal` | Standard complete run. | Recommended for production. |
| `debug` | Uses a special reduced/sample branch. | Not a calibrated full-particle run. |
| `continue` | Reloads existing particle tables and counts, skips cube FITS files already present. | Requires a previous run with `write_table: true`; unavailable for `DEFAULT` instrument mode. |

`other_params.ncpu` controls multiprocessing workers; `null` selects `multiprocessing.cpu_count()`. `other_params.nprocess_ploop` sets the batch size of scheduled nonempty spaxels. Large SSP grids, broad wavelength coverage and many workers may increase memory requirements.

## Interpretation checks

The present code uses a particular mass-normalisation convention: the standard binning branch assigns particle mass weights of one, followed by optional global multiplication by `gal_mass / nparticles`. Validate your catalogue, template normalisation and distance scaling before using cube fluxes quantitatively. The code is not a generic input-data converter.
