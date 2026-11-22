# Output files and formats

Outputs are written in `<outputDir>/<configName>/`.

| File | Description |
| --- | --- |
| `data_cube_<i>.fits` | Synthetic IFS cube, wavelength axis and (if enabled) uncertainty image. |
| `mass_fraction_array_<i>.npy` | SSP population-weight distribution across spatial bins. |
| `statistic_count<i>.npy` | Number of particles in each spatial bin. |
| `particle_table_<i>.fits` | Intermediate catalogue for a cube; created with `write_table: true`. |
| `grids_distrib_<i>.png` | Maps of selected stellar-population quantities. |
| `datacube_distrib.png` | Diagram of cube footprints over the particle sky distribution. |
| `cube_list` | Saved list of sky-coordinate cube centres. |
| `setup.json` | Effective configuration saved after processing. |
| `outputs.log` | Progress and diagnostic messages. |

## FITS extensions

- **Primary HDU:** metadata about the model, galaxy placement, instrument settings, noise, SSP library, and GalCraft version.
- **`DATA`:** synthetic flux. `CubeMaker.write_cube` transposes the in-memory `data_cube[x, y, wavelength]` to FITS order `(wavelength, y, x)` as read by Astropy. The header sets `BUNIT = 10**(-20)*erg/s/cm**2/Angstrom`.
- **`ERROR`:** present only when `add_noise: true`. Contains the *standard deviation used to sample Gaussian noise*, in the same units and shape as `DATA`.
- **`WAVE`:** one-dimensional wavelength values in Angstrom.

```python
from astropy.io import fits
import numpy as np

with fits.open('output/test_kit/data_cube_0.fits') as h:
    data = h['DATA'].data
    wavelength = h['WAVE'].data
    error = h['ERROR'].data if 'ERROR' in h else None

print('cube:', data.shape)
print('wavelength interval:', wavelength[0], wavelength[-1], 'Angstrom')
print('median spectrum:', np.nanmedian(data, axis=(1, 2)).shape)
```

The `ERROR` extension is not a measured instrumental variance model; its values follow the simplified formula implemented in the cube writer. Negative/invalid input fluxes could make the square-root noise model inappropriate. Use the `WAVE` extension as the authoritative spectral coordinate array.

## Population weights

`mass_fraction_array_<i>.npy` is stored after transposing the SSP-age, metallicity and alpha dimensions and the spatial axes. Its array dimensions are `(n_age, n_metal, n_alpha, n_y, n_x)` according to the spectrum-generator code; check these dimensions against your specific template grid before deriving physical maps.
