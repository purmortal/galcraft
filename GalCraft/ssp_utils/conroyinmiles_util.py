import glob, os
import numpy as np


def age_metal_alpha(filename, age_grid, metal_grid, alpha_grid):
    '''
    extract the age and metallicity from the name of file
    modified from miles_util.py of PPXF
    '''
    s = filename[:-4].split('_')
    age = age_grid[int(s[-2])]
    metal = metal_grid[int(s[-1])]
    alpha = alpha_grid[int(os.path.dirname(filename).split('/')[-1].split('_')[-1])]
    return age, metal, alpha


class ssp:
    '''
    This is a class which is written for uploading the pegase-hr SSP models.
    The SSP models have been interpolated in a better grid, following the steps
    in Kacharov et al. 2018 using ulyss.
    The new age and metallicity grid is similar as the MILES SSP models.
    This file is modified from miles_util.py of PPXF
    '''
    def __init__(self, path_library, FWHM_tem=2.51,
                 age_range=None, metal_range=None, alpha_range=None):

        files = glob.glob(path_library)
        assert len(files) > 0, "Files not found %s" % path_library

        # Read data
        ssp = np.load(files[0])
        lam = np.load(os.path.dirname(os.path.dirname(path_library)) + '/wave.npy')

        # Extract ages, metallicities and alpha from the templates
        ages = np.load(os.path.dirname(os.path.dirname(files[0])) + '/logage_grid.npy')
        ages = 10**(ages)
        metals = np.load(os.path.dirname(os.path.dirname(files[0])) + '/metal_grid.npy')
        alphas = np.load(os.path.dirname(os.path.dirname(files[0])) + '/alpha_grid.npy')
        all = [age_metal_alpha(f, ages, metals, alphas) for f in files]
        nAges, nMetal, nAlpha = len(ages), len(metals), len(alphas)

        # Arrays to store templates
        templates          = np.zeros((ssp.size, nAges, nMetal, nAlpha))
        templates[:,:,:,:] = np.nan

        # Arrays to store properties of the models
        age_grid    = np.empty((nAges, nMetal, nAlpha))
        metal_grid  = np.empty((nAges, nMetal, nAlpha))
        alpha_grid  = np.empty((nAges, nMetal, nAlpha))

        # This sorts for alphas
        for i, alpha in enumerate(alphas):
            # This sorts for ages
            for j, age in enumerate(ages):
                # This sorts for metals
                for k, mh in enumerate(metals):
                    p = all.index((age, mh, alpha))
                    ssp = np.load(files[p])
                    age_grid[j, k, i] = age
                    metal_grid[j, k, i] = mh
                    alpha_grid[j, k, i] = alpha
                    templates[:, j, k, i] = ssp

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
