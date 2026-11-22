# Getting started

## Install

Install the published package:

```bash
python -m pip install GalCraft
```

Or work from a clone:

```bash
git clone https://github.com/purmortal/galcraft.git
cd galcraft
python -m pip install .
```

The package declares the dependencies `astropy`, `numpy`, `scipy`, `matplotlib`, `ebfpy`, `ephem` and `spectres`. The installed console command is `GalCraft` (capital G and C).

## First run: official test kit

The example input files live in a *different branch* from the main package source. Download and extract the test kit:

```bash
curl -L https://github.com/purmortal/galcraft/archive/refs/heads/test_kit.zip -o test_kit.zip
unzip test_kit.zip
cd galcraft-test_kit/tests
GalCraft --config test_kit --default-dir configFiles/defaultDir
```

The test-kit configuration uses a 10 × 10 mock cube, 2-arcsec pixels, an MILES SSP library, and a wavelength range of 4500–4700 Å. It disables added noise and extinction, but enables particle line-of-sight velocities.

The provided example writes to `output/test_kit/`, relative to the test kit's working-directory layout. Expect files such as `data_cube_0.fits`, `mass_fraction_array_0.npy`, diagnostic plots, and `outputs.log`.

:::{important}
`--config test_kit` is a *configuration name*, not a JSON pathname. GalCraft opens `setup_test_kit.json` in the `configDir` configured by `--default-dir`. Non-DEFAULT instrument modes also look for the corresponding `setup_test_kit_list` of cube centres in that directory.
:::

## Inspect a resulting cube

```python
from astropy.io import fits

with fits.open("output/test_kit/data_cube_0.fits") as hdul:
    hdul.info()
    flux = hdul["DATA"].data  # (wavelength, y, x)
    wave = hdul["WAVE"].data  # Angstrom
    print(flux.shape, wave[[0, -1]])
```

The `ERROR` image extension is only included when `add_noise` is enabled. Read the [outputs](outputs.md) page before interpreting units or auxiliary arrays.

## Command-line options

```text
GalCraft --config NAME --default-dir PATH
```

| Option | Meaning |
| --- | --- |
| `--config NAME` | Selects `setup_NAME.json` and the output subdirectory `NAME/`. |
| `--default-dir PATH` | Text file defining `configDir`, `modelDir`, `templateDir`, and `outputDir`. |

GalCraft's current CLI parser does not accept a JSON config filepath directly. Both options should be supplied.
