# Getting started

## Installation

Install the distributed package with pip:

```bash
python -m pip install GalCraft
```

Alternatively, install a local checkout (useful while changing the code or documentation):

```bash
git clone https://github.com/purmortal/galcraft.git
cd galcraft
python -m pip install -e .
```

GalCraft's package dependencies are `astropy`, `numpy`, `scipy`, `matplotlib`, `ebfpy`, `ephem` and `spectres`. Its executable command is **`GalCraft`** (capital G and C).

## Run the official example

The test kit is maintained in a separate repository branch and contains its own working directories, configuration and inputs:

```bash
curl -L https://github.com/purmortal/galcraft/archive/refs/heads/test_kit.zip -o test_kit.zip
unzip test_kit.zip
cd galcraft-test_kit/tests
GalCraft --config test_kit --default-dir configFiles/defaultDir
```

The example configuration requests a 10 × 10 cube with 2-arcsec spatial pixels, MILES SSP templates, and a final 4500–4700 Å wavelength window. It enables stellar line-of-sight velocities but disables simulated noise and extinction.

:::{important}
The value supplied to `--config` is the **configuration name**, not a JSON file path. In this example, the program opens `setup_test_kit.json` inside the configured `configDir` and writes results to `<outputDir>/test_kit/`.
:::

## Command-line options

```text
GalCraft --config NAME --default-dir PATH
```

| Argument | Meaning |
| --- | --- |
| `--config NAME` | Base name of `setup_NAME.json`, and name of the output subdirectory. |
| `--default-dir PATH` | Path to a text file mapping `configDir`, `modelDir`, `templateDir` and `outputDir`. |

In the current implementation, the command does not accept a JSON filename as a substitute for `--config NAME`.

## Inspect the generated FITS cube

```python
from astropy.io import fits
import numpy as np

with fits.open('output/test_kit/data_cube_0.fits') as hdul:
    hdul.info()
    flux = hdul['DATA'].data       # (wavelength, y, x)
    wave = hdul['WAVE'].data       # Angstrom
    noise_sigma = hdul['ERROR'].data if 'ERROR' in hdul else None

print('Data shape:', flux.shape)
print('Wavelength limits:', wave[0], wave[-1])
print('Median spectrum:', np.nanmedian(flux, axis=(1, 2)).shape)
```

An `ERROR` extension exists **only** when `cube_params.add_noise` is `true`. See [Outputs](outputs.md) for all generated files and their conventions.

## Your own simulation

1. Prepare an EBF particle catalogue with the required physical and coordinate fields.
2. Obtain SSP spectra compatible with your chosen [SSP reader](configuration.md#ssp-reader-and-input-data).
3. Create the four-directory mapping file and `setup_NAME.json`.
4. For `DIY` or an instrument preset, create `setup_NAME_list` containing one `x,y` centre per row.
5. Run `GalCraft --config NAME --default-dir /path/to/defaultDir`.
6. Check `outputs.log`, `setup.json`, diagnostic maps and FITS wavelength axes before interpreting fluxes.

The [Workflow](workflow.md) explains every processing stage, and [Configuration](configuration.md) describes the supported keys and assumptions.
