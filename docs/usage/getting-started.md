# Getting started

## Installation

Install GalCraft with pip:

```bash
pip install GalCraft
```

Install from the git repo:

```bash
git clone https://github.com/purmortal/galcraft.git
cd galcraft
pip install .
```

GalCraft's package dependencies are `astropy`, `numpy`, `scipy`, `matplotlib`, `ebfpy`, `ephem` and `spectres`. Its executable command is **`GalCraft`** (capital G and C).

## Running GalCraft

GalCraft is executed from the command line using:

```bash
GalCraft --config NAME --default-dir PATH
```

where:

| Argument | Description                                                                                                                                                       |
| --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config NAME` | Name of the run. GalCraft loads the corresponding `setup_NAME.json` file from `configDir` and saves the outputs in a subdirectory named `NAME` under `outputDir`. |
| `--default-dir PATH` | Path to a text file defining the four directories used by GalCraft: `configDir`, `modelDir`, `templateDir`, and `outputDir`.                                      |

**Note:** The `--config` argument accepts the configuration name (`NAME`), rather than the full JSON filename (`setup_NAME.json`) or its file path.


## Generate your first data cube

You can generate your first GalCraft data cube using the provided test kit:

```bash
wget https://github.com/purmortal/galcraft/archive/refs/heads/test_kit.zip
unzip test_kit.zip
cd galcraft-test_kit/tests/
GalCraft --config test_kit --default-dir configFiles/defaultDir
```

This will generate a 10 × 10 spatial-pixel data cube with a pixel scale of 2 arcsec, using MILES SSP templates over the wavelength range of 4500–4700 Å. 
The resulting data cube does not include noise or extinction effects. 
All output files are saved in `<outputDir>/test_kit/`.



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

An `ERROR` extension exists **only** when `cube_params.add_noise` is `true`. 
See [Input & Output](input-output.md#output-files) for all generated files and their conventions.

## Generate your own mock data

To generate custom mock IFS data cubes with GalCraft, follow these steps:

1. **Prepare the stellar particle catalogue:** Prepare a stellar particle catalogue containing the required stellar population properties, 3D velocities and spatial coordinates. The current command-line reader expects an EBF file; see [Input files](input-output.md#input-files) for details.
2. **Prepare the SSP templates:** Obtain the SSP templates supported by your chosen [SSP configuration](configuration.md#ssp-spectra-ssp_params).
3. **Set up the directories and configuration:** Create a `defaultDir` file specifying the four working directories, and a `setup_NAME.json` file defining the GalCraft configurations.
4. **Define the cube pointings:** Create a `setup_NAME_list` file specifying the centre coordinates (`x, y`) of each data cube, with one pointing per row.
5. **Run GalCraft** using:
   ```bash
   GalCraft --config NAME --default-dir /path/to/defaultDir
   ```
6. **Inspect the results:** Check the generated FITS data cubes and other output files in the corresponding output directory.  See [Input & Output](input-output.md#output-files) for the file inventory.
