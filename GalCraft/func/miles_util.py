from os import path
import glob, re

import numpy as np
from astropy.io import fits

import ppxf.ppxf_util as util




def age_metal(filename):
    """
    Extract the age and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2022

    This function relies on the MILES file containing a substring of the
    precise form like Zm0.40T00.0794, specifying the metallicity and age.

    :param filename: string possibly including full path
        (e.g. 'miles_library/Eun1.30Zm0.40T00.0794.fits')
    :return: age (Gyr), [M/H]

    """
    # s = re.findall(r'Z[m|p][0-9]\.[0-9]{2}T[0-9]{2}\.[0-9]{4}', filename)[0]
    s = re.findall(r'Z[m|p]\d+\.\d+T\d+\.\d+', filename)[0]
    split_result = s.split("T")
    metal = split_result[0]
    age = float(split_result[1])
    if "Zm" in metal:
        metal = -float(metal[2:])
    elif "Zp" in metal:
        metal = float(metal[2:])

    return age, metal




class miles:


    def __init__(self, pathname, FWHM_tem=2.51, age_range=None,
                 metal_range=None):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the galaxy spectrum, to determine the
        # size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam = h2['CRVAL1'] + np.arange(h2['NAXIS1']) * h2['CDELT1']
        # lam_range_temp = lam[[0, -1]]
        # ssp_new, ln_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]
        # ssp_new = ssp

        # if norm_range is not None:
        #     norm_range = np.log(norm_range)
        #     band = (norm_range[0] <= lam) & (lam <= norm_range[1])

        templates = np.empty((ssp.size, n_ages, n_metal))
        age_grid, metal_grid, flux = np.empty((3, n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        # if FWHM_gal is not None:
        #     FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
        #     sigma = FWHM_dif / 2.355 / h2['CDELT1']  # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, met in enumerate(metals):
                p = all.index((age, met))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                # if FWHM_gal is not None:
                #     if np.isscalar(FWHM_gal):
                #         if sigma > 0.1:  # Skip convolution for nearly zero sigma
                #             ssp = ndimage.gaussian_filter1d(ssp, sigma)
                #     else:
                #         ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                # ssp_new = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                # ssp_new = ssp
                # if norm_range is not None:
                #     flux[j, k] = np.mean(ssp_new[band])
                #     ssp_new /= flux[j, k]  # Normalize every spectrum
                templates[:, j, k] = ssp
                age_grid[j, k] = age
                metal_grid[j, k] = met

        if age_range is not None:
            w = (age_range[0] <= age_grid[:, 0]) & (age_grid[:, 0] <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            flux = flux[w, :]
            n_ages, n_metal = age_grid.shape

        if metal_range is not None:
            w = (metal_range[0] <= metal_grid[0, :]) & (metal_grid[0, :] <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            flux = flux[:, w]
            n_ages, n_metal = age_grid.shape

        # The code below is the cause of the flux scale issue!!!!!
        # if norm_range is None:
        #     flux = np.median(templates[templates > 0])
        #     templates /= flux  # Normalize by a scalar


        self.templates = templates
        self.lam_temp = lam
        if 'loginterp' in pathname:
            self.age_grid = 10 ** (age_grid - 9)
        else:
            self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.flux = flux
        self.FWHM_tem = FWHM_tem

    ###############################################################################
    # MODIFICATION HISTORY:
    #   V1.0.0: Written. Michele Cappellari, Oxford, 1 December 2016
    #   V1.0.1: Use path.realpath() to deal with symbolic links.
    #       Thanks to Sam Vaughan (Oxford) for reporting problems.
    #       MC, Garching, 11 January 2016
    #   V1.0.2: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
    #   V1.0.3: Removed dependency on cap_readcol. MC, Oxford, 10 May 2018

    def mass_to_light(self, weights, band="r", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced
        in output by pPXF. A Salpeter IMF is assumed (slope=1.3).
        The returned M/L includes living stars and stellar remnants,
        but excludes the gas lost during stellar evolution.

        This procedure uses the photometric predictions
        from Vazdekis+12 and Ricciardelli+12
        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R
        I downloaded them from http://miles.iac.es/ in December 2016 and I
        included them in pPXF with permission.

        :param weights: pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
        :param band: possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
            the Vega photometric system and "u", "g", "r", "i" for the SDSS AB system.
        :param quiet: set to True to suppress the printed output.
        :return: mass_to_light in the given band

        """
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        sdss_bands = ["u", "g", "r", "i"]
        vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
        sdss_sun_mag = [6.55, 5.12, 4.68, 4.57]  # values provided by Elena Ricciardelli

        ppxf_dir = path.dirname(path.realpath(util.__file__))

        if band in vega_bands:
            k = vega_bands.index(band)
            sun_mag = vega_sun_mag[k]
            file2 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
        elif band in sdss_bands:
            k = sdss_bands.index(band)
            sun_mag = sdss_sun_mag[k]
            file2 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        else:
            raise ValueError("Unsupported photometric band")

        file1 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_mass_Padova00_UN_baseFe_v10.0.txt"
        slope1, MH1, Age1, m_no_gas = np.loadtxt(file1, usecols=[1, 2, 3, 5]).T

        slope2, MH2, Age2, mag = np.loadtxt(file2, usecols=[1, 2, 3, 4 + k]).T

        # The following loop is a brute force, but very safe and general,
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mass_no_gas_grid = np.empty_like(weights)
        lum_grid = np.empty_like(weights)
        for j in range(self.n_ages):
            for k in range(self.n_metal):
                p1 = (np.abs(self.age_grid[j, k] - Age1) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH1) < 0.01) & \
                     (np.abs(1.30 - slope1) < 0.01)  # Salpeter IMF
                mass_no_gas_grid[j, k] = m_no_gas[p1]

                p2 = (np.abs(self.age_grid[j, k] - Age2) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH2) < 0.01) & \
                     (np.abs(1.30 - slope2) < 0.01)  # Salpeter IMF
                lum_grid[j, k] = 10 ** (-0.4 * (mag[p2] - sun_mag))

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights * mass_no_gas_grid) / np.sum(weights * lum_grid)

        if not quiet:
            print(f'(M*/L)_{band}: {mlpop:#.4g}')

        return mlpop



    def plot(self, weights, nodots=False, colorbar=True, **kwargs):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(xgrid, ygrid, weights,
                             nodots=nodots, colorbar=colorbar, **kwargs)

    ##############################################################################

    def mean_age_metal(self, weights, quiet=False):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        lg_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_lg_age = np.sum(weights * lg_age_grid) / np.sum(weights)
        mean_metal = np.sum(weights * metal_grid) / np.sum(weights)

        if not quiet:
            print('Weighted <lg_age> [yr]: %#.3g' % mean_lg_age)
            print('Weighted <[M/H]>: %#.3g' % mean_metal)

        return mean_lg_age, mean_metal

##############################################################################
