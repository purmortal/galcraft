import glob, re
import numpy as np
from astropy.io import fits




def age_metal_alpha(passedFiles):
    """
    Function to extract the values of age, metallicity, and alpha-enhancement
    from standard MILES filenames. Note that this function can automatically
    distinguish between template libraries that do or do not include
    alpha-enhancement.
    """

    out = np.zeros((len(passedFiles),3)); out[:,:] = np.nan

    files = []
    for i in range( len(passedFiles) ):
        files.append( passedFiles[i].split('/')[-1] )

    for num, s in enumerate(files):
        selected_s = re.findall(r'Z[m|p]\d+\.\d+T\d+\.\d+', s)[0]
        split_result = selected_s.split("T")
        age = float(split_result[1])
        s_metal = split_result[0]
        if "Zm" in s_metal:
            metal = -float(s_metal[2:])
        elif "Zp" in s_metal:
            metal = float(s_metal[2:])
        else:
            raise ValueError("             This is not a standard MILES filename")

        # Alpha
        if s.find('baseFe') == -1:
            EMILES = False
        elif s.find('baseFe') != -1:
            EMILES = True

        if EMILES == False:
            # Usage of MILES: There is a alpha defined
            e = s.find('E')
            alpha = float( s[e+2 : e+6] )
        elif EMILES == True:
            # Usage of EMILES: There is *NO* alpha defined
            alpha = 0.0

        out[num,:] = age, metal, alpha

    Age   = np.unique( out[:,0] )
    Metal = np.unique( out[:,1] )
    Alpha = np.unique( out[:,2] )
    nAges  = len(Age)
    nMetal = len(Metal)
    nAlpha = len(Alpha)

    float_metal_str = '{:.' + str(len(s_metal[2:])-2) + 'f}'
    metal_str = []
    alpha_str = []
    for i in range( len(Metal) ):
        if Metal[i] > 0:
            mm = 'p'+float_metal_str.format(np.abs(Metal[i]))+'T'
        elif Metal[i] < 0:
            mm = 'm'+float_metal_str.format(np.abs(Metal[i]))+'T'
        metal_str.append(mm)
    for i in range( len(Alpha) ):
        if EMILES == False:
            alpha_str.append( 'Ep'+'{:.2f}'.format(Alpha[i]) )
        elif EMILES == True:
            alpha_str = ['baseFe']

    if 'loginterp' in passedFiles[0]:
        Age = 10 ** (Age - 9)

    return( Age, Metal, Alpha, metal_str, alpha_str, nAges, nMetal, nAlpha )




class ssp:


    def __init__(self, path_library, FWHM_tem=2.51,
                 age_range=None, metal_range=None, alpha_range=None):

        sp_models = glob.glob(path_library)
        assert len(sp_models) > 0, "Files not found %s" % path_library
        sp_models.sort()

        # Read data
        hdu_spmod      = fits.open(sp_models[0])
        ssp_data       = hdu_spmod[0].data
        ssp_head       = hdu_spmod[0].header
        lam            = ssp_head['CRVAL1'] + np.arange(ssp_head['NAXIS1']) * ssp_head['CDELT1']

        # Extract ages, metallicities and alpha from the templates
        ages, metals, alphas, metal_str, alpha_str, nAges, nMetal, nAlpha = age_metal_alpha(sp_models) # revised by Zixian Wang

        # Arrays to store templates
        templates          = np.zeros((ssp_data.size, nAges, nMetal, nAlpha))
        templates[:,:,:,:] = np.nan

        # Arrays to store properties of the models
        age_grid    = np.empty((nAges, nMetal, nAlpha))
        metal_grid  = np.empty((nAges, nMetal, nAlpha))
        alpha_grid  = np.empty((nAges, nMetal, nAlpha))

        for i, age in enumerate(alpha_str):
            # This sorts for metals
            for k, mh in enumerate(metal_str):
                files = [s for s in sp_models if (mh in s and age in s)]
                # This sorts for ages
                for j, filename in enumerate(files):
                    hdu = fits.open(filename)
                    ssp = hdu[0].data
                    age_grid[j, k, i]    = ages[j]
                    metal_grid[j, k, i]  = metals[k]
                    alpha_grid[j, k, i]  = alphas[i]
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
