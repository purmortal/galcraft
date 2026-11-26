# Generate your own data cube

I recommend starting with the [official GalCraft test-kit configuration](https://github.com/purmortal/galcraft/tree/test_kit/tests/configFiles) before creating your own run.

To create a custom mock IFS data cube, follow these steps:

1. **Prepare the stellar-particle catalogue.** Supply the required particle properties (age, metallicity and [α/Fe]) together with 3D spatial coordinates and velocities; include mass weights, distances and extinction where needed. The current CLI expects **EBF** input. Convert other simulation outputs to this format or adapt the source reader.
2. **Prepare the SSP templates.** Obtain an SSP library and arrange its files for your chosen [SSP reader](configuration.md#ssp-reader-and-input-data).
3. **Configure directories and simulation parameters.** Create a `defaultDir` file pointing to `configDir`, `modelDir`, `templateDir` and `outputDir`, plus a `setup_NAME.json` file with the desired options.
4. **Define spatial pointings.** For `DIY` or an instrument preset, create `setup_NAME_list` with one comma-separated pair of centre coordinates (`x,y`, in degrees) on each line, matching the configured coordinate system.
5. **Run GalCraft.** From a terminal, execute:

   ```bash
   GalCraft --config NAME --default-dir /path/to/defaultDir
   ```

6. **Inspect the results.** Review `outputs.log`, `setup.json`, the FITS cubes, NumPy arrays and diagnostic plots in `<outputDir>/NAME/`.

```{note}
Start from the [official test-kit configuration](https://github.com/purmortal/galcraft/tree/test_kit/tests/configFiles). When adapting a different simulation, check SSP units, model mass calibration, spatial orientation and whether extinction, line-of-sight velocities or noise are enabled.
```


