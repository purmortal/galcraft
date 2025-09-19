import glob
import os
import numpy as np
from astropy.io import fits
from GalCraft.modules.constant import *


def vactoair(vacwl):
    """Calculate the approximate wavelength in air for vacuum wavelengths.

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses an approximate formula from the IDL astronomy library
    https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro

    """
    wave2 = vacwl * vacwl
    n = 1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2 * wave2)

    # Do not extrapolate to very short wavelengths.
    if not isinstance(vacwl, np.ndarray):
        if vacwl < 2000:
            n = 1.0
    else:
        ignore = np.where(vacwl < 2000)
        n[ignore] = 1.0

    return vacwl / n



class ssp:

    def __init__(self, path_library, FWHM_tem='lsf_alphaMC',
                 age_range=None, metal_range=None, alpha_range=None):

        # SSP model library
        sp_models = glob.glob(path_library)
        assert len(sp_models) > 0, "Files not found %s" % path_library

        # Read data
        ssp_alphaMC = fits.open(sp_models[0])
        lam = ssp_alphaMC[0].data
        orig_templates = ssp_alphaMC[1].data
        grids = ssp_alphaMC[2].data

        # transfer from vacuum to air wave
        lam_air = vactoair(lam)
        for j in range(len(orig_templates)):
            orig_templates[j, :] = np.interp(lam, lam_air, orig_templates[j, :])

        # Select wavelength between 3000 and 10000 Angstrom (should change to air wave)
        hires_mask = (lam > vactoair(3000)) & (lam < vactoair(10000))
        lam = lam[hires_mask]
        orig_templates = orig_templates[:, hires_mask]
        ssp_data = orig_templates[0, :]

        # Change the unit of templates to from dHz to dlambda
        orig_templates = orig_templates * cvel * 1e13 / lam ** 2

        # Extract ages, metallicities and alpha from the templates
        orig_age_grid   = 10 ** (np.array([x[0] for x in grids]) - 9)
        orig_metal_grid = np.array([x[1] for x in grids])
        orig_alpha_grid = np.array([x[2] for x in grids])
        ages   = np.unique(orig_age_grid)
        metals = np.unique(orig_metal_grid)
        alphas = np.unique(orig_alpha_grid)
        nAges, nMetal, nAlpha = len(ages), len(metals), len(alphas)

        # Arrays to store templates
        templates          = np.zeros((ssp_data.size, nAges, nMetal, nAlpha))
        templates[:,:,:,:] = np.nan

        # Arrays to store properties of the models
        age_grid    = np.empty((nAges, nMetal, nAlpha))
        metal_grid  = np.empty((nAges, nMetal, nAlpha))
        alpha_grid  = np.empty((nAges, nMetal, nAlpha))

        # Load the templates
        for j in range(len(orig_templates)):
            idx_j = np.where(ages   == orig_age_grid[j])[0][0]
            idx_k = np.where(metals == orig_metal_grid[j])[0][0]
            idx_i = np.where(alphas == orig_alpha_grid[j])[0][0]
            ssp = orig_templates[j, :]
            age_grid[idx_j, idx_k, idx_i]   = orig_age_grid[j]
            metal_grid[idx_j, idx_k, idx_i] = orig_metal_grid[j]
            alpha_grid[idx_j, idx_k, idx_i] = orig_alpha_grid[j]
            templates[:, idx_j, idx_k, idx_i] = ssp

        # Interpolate templates into linear wave grid
        new_wave = np.arange(lam[0], lam[-1], np.min(np.diff(lam)))
        new_templates = np.zeros([len(new_wave), templates.shape[1], templates.shape[2], templates.shape[3]])
        for j in range(templates.shape[1]):
            for k in range(templates.shape[2]):
                for i in range(templates.shape[3]):
                    new_templates[:, j, k, i] = np.interp(new_wave, lam, templates[:, j, k, i])

        lam = new_wave
        templates = new_templates
        new_wave = None
        new_templates = None

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
