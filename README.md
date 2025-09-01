
# GalCraft: Building integral-field spectrograph data cubes of the Milky Way


[![pypi](https://img.shields.io/badge/python-pypi-blue.svg)](https://pypi.org/project/GalCraft/)
[![arXiv](https://img.shields.io/badge/arxiv-2310.18258-b31b1b.svg)](https://arxiv.org/abs/2310.18258)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fmnras%2Fstae2148-green.svg)](https://doi.org/10.1093/mnras/stae2148)
[![ASCL](https://img.shields.io/badge/ASCL-2410.001-%23262255.svg)](https://ascl.net/2410.001)
[![LICENSE](https://img.shields.io/badge/lisence-MIT-blue.svg)](https://github.com/purmortal/galcraft/blob/main/LICENSE)

GalCraft is a flexible software to create mock IFS observations of the Milky Way and other hydrodynamical/N-body simulations. It is entirely written in Python3 and conducts all the procedures from inputting data and spectral templates to the output of IFS data cubes in `fits` format. 

The produced mock data cubes can be analyzed in the same way as real IFS observations by many methods, particularly codes like Voronoi binning ([Cappellari & Copin 2003](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C/abstract)), Penalized Pixel-Fitting (pPXF, [Cappellari & Emsellem 2004](https://ui.adsabs.harvard.edu/abs/2004PASP..116..138C/abstract); [Cappellari 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract), [2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C/abstract)), line-strength indices (e.g., [Worthey 1994](https://ui.adsabs.harvard.edu/abs/1994ApJS...95..107W/abstract); [Schiavon 2007](https://ui.adsabs.harvard.edu/abs/2007ApJS..171..146S/abstract); [Thomas et al. 2011](https://ui.adsabs.harvard.edu/abs/2011MNRAS.412.2183T/abstract); [Martín-Navarro et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.3700M/abstract)), or a combination of them (e.g., the GIST pipeline, [Bittner et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...628A.117B/abstract)).

An elaborate, Python-native parallelization is implemented and tested on various machines from laptops to cluster scales.



## Installation

### Using pip

```
pip install GalCraft
```

### From the git repo

```
git clone https://github.com/purmortal/galcraft.git
cd galcraft
pip install .
```

## Example

Follow the commands below to download test kit:

```
wget https://github.com/purmortal/galcraft/archive/refs/heads/test_kit.zip
unzip test_kit.zip
cd galcraft-test_kit/tests/
```

Then generate your first mock data using GalCraft command:
```
GalCraft --config test_kit --default-dir configFiles/defaultDir
```

The data cubes are saved into `./output/test_kit/`

## Documentation
A detailed documentation of GalCraft will be available soon.




## Citing GalCraft
If you use this software framework for any publication, please cite the original paper [Wang et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024MNRAS.534.1175W/abstract), which describes the method and its application to mock Milky Way observations.

```
@ARTICLE{2024MNRAS.534.1175W,
       author = {{Wang}, Zixian and {Sharma}, Sanjib and {Hayden}, Michael R. and {van de Sande}, Jesse and {Bland-Hawthorn}, Joss and {Vaughan}, Sam and {Martig}, Marie and {Pinna}, Francesca},
        title = "{Validating full-spectrum fitting with a synthetic integral-field spectroscopic observation of the Milky Way}",
      journal = {\mnras},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = oct,
       volume = {534},
       number = {2},
        pages = {1175-1204},
          doi = {10.1093/mnras/stae2148},
archivePrefix = {arXiv},
       eprint = {2310.18258},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024MNRAS.534.1175W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```




## License
This software is governed by the MIT License. In brief, you can use, distribute, and change this package as you want.


## Contact 
- Zixian Wang (wang.zixian.astro@gmail.com; zixian.wang@utah.edu)
- Michael Hayden (mrhayden@ou.edu)
- Sanjib Sharma (ssharma@stsci.edu)
