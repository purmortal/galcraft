import glob, os
import numpy as np


def age_metal(filename, age_grid, metal_grid):
    '''
    extract the age and metallicity from the name of file
    modified from miles_util.py of PPXF
    '''
    s = filename[:-4].split('_')
    age = age_grid[int(s[-2])]
    metal = metal_grid[int(s[-1])]
    return age, metal



class ssp():
    '''
    This is a class which is written for uploading the pegase-hr SSP models.
    The SSP models have been interpolated in a better grid, following the steps
    in Kacharov et al. 2018 using ulyss.
    The new age and metallicity grid is similar as the MILES SSP models.
    This file is modified from miles_util.py of PPXF
    '''
    def __init__(self, path_library, FWHM_tem=0.55,
                 age_range=None, metal_range=None, alpha_range=None):

        files = glob.glob(path_library)
        assert len(files) > 0, "Files not found %s" % path_library

        # Read data
        ssp = np.genfromtxt(files[0], delimiter=',')[1:]
        lam = (3900 + np.arange(len(ssp)+1)*0.2)[1:]

        # Extract ages, metallicities and alpha from the templates
        ages   = np.genfromtxt(os.path.dirname(files[0]) + '/log10age_grid.csv',delimiter=',')
        ages   = 10 ** (ages - 6) / 1000
        metals = np.genfromtxt(os.path.dirname(files[0]) + '/feh_grid.csv',delimiter=',')
        alphas  = np.array([0])
        all = [age_metal(f, ages, metals) for f in files]
        nAges, nMetal, nAlpha = len(ages), len(metals), len(alphas)

        # Arrays to store templates
        templates          = np.zeros((ssp.size, nAges, nMetal, nAlpha))
        templates[:,:,:,:] = np.nan

        # Arrays to store properties of the models
        age_grid    = np.empty((nAges, nMetal, nAlpha))
        metal_grid  = np.empty((nAges, nMetal, nAlpha))
        alpha_grid  = np.empty((nAges, nMetal, nAlpha))

        # This sorts for ages
        for j, age in enumerate(ages):
            # This sorts for metals
            for k, mh in enumerate(metals):
                p = all.index((age, mh))
                ssp = np.genfromtxt(files[p], delimiter=',')[1:]
                ssp = ssp * 1e-7 # Convert the SSP unit to MILES unit to make them the same
                age_grid[j, k, 0] = age
                metal_grid[j, k, 0] = mh
                alpha_grid[j, k, 0] = alphas[0]
                templates[:, j, k, 0] = ssp

        if age_range is not None:
            w = (age_range[0] <= age_grid[:, 0, 0]) & (age_grid[:, 0, 0] <= age_range[1])
            templates = templates[:, w, :, :]
            age_grid = age_grid[w, :, :]
            metal_grid = metal_grid[w, :, :]
            alpha_grid = alpha_grid[w, :, :]
            nAges, nMetal, nAlpha = age_grid.shape

        if metal_range is not None:
            w = (metal_range[0] <= metal_grid[0, :, 0]) & (metal_grid[0, :, 0] <= metal_range[1])
            templates = templates[:, :, w, :]
            age_grid = age_grid[:, w, :]
            metal_grid = metal_grid[:, w, :]
            alpha_grid = alpha_grid[:, w, :]
            nAges, nMetal, nAlpha = age_grid.shape

        if alpha_range is not None:
            w = (alpha_range[0] <= alpha_grid[0, 0, :]) & (alpha_grid[0, 0, :] <= alpha_range[1])
            templates = templates[:, :, :, w]
            age_grid = age_grid[:, :, w]
            metal_grid = metal_grid[:, :, w]
            alpha_grid = alpha_grid[:, :, w]
            nAges, nMetal, nAlpha = age_grid.shape

        self.templates = templates
        self.lam_temp = lam
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.alpha_grid = alpha_grid
        self.n_ages = nAges
        self.n_metal = nMetal
        self.n_alpha = nAlpha
        self.FWHM_tem = FWHM_tem
