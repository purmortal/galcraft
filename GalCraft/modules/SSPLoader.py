
import os
import logging
import numpy as np
import importlib.util
from scipy import ndimage
from spectres import spectres

from multiprocessing import Pool
import GalCraft.modules.utils as utils
from GalCraft.modules.constant import *


class model:

    def __init__(self, templateDir, instrumentDir, ssp_params, other_params):

        # setup some ssp_params
        self.ssp_name = ssp_params['model']
        self.library = ssp_params['library']
        self.factor = ssp_params['factor']
        self.FWHM_gal = ssp_params['FWHM_gal']
        self.FWHM_tem = ssp_params['FWHM_tem']
        self.dlam = ssp_params['dlam']
        self.age_range = ssp_params['age_range']
        self.metal_range = ssp_params['metal_range']
        self.alpha_range = ssp_params['alpha_range']
        self.wave_range = ssp_params['wave_range']
        self.interpolator_method = ssp_params['spec_interpolator']
        self.ncpu = other_params['ncpu']


        # Load SSP models
        model = self.load_ssp(templateDir=templateDir)
        templates = model.templates


        # Define some constants from SSP parameters
        reg_dim = templates.shape[1:]
        wave = model.lam_temp
        velscale = cvel * np.log(wave[1]/wave[0])
        age_grid = model.age_grid[:, 0, 0]
        metal_grid = model.metal_grid[0, :, 0]
        alpha_grid = model.alpha_grid[0, 0, :]
        age_grid_3d = model.age_grid
        metal_grid_3d = model.metal_grid
        alpha_grid_3d = model.alpha_grid

        self.FWHM_tem = model.FWHM_tem


        # This is for calculating the bins in age, metallicity
        xgrid = np.log10(age_grid) + 9
        ygrid = metal_grid
        x = xgrid
        y = ygrid
        xb = (x[1:] + x[:-1])/2  # internal grid borders
        yb = (y[1:] + y[:-1])/2
        xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])  # 1st/last border
        yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
        if len(alpha_grid) > 1:
            self.single_alpha = False
            zgrid = alpha_grid
            z = zgrid
            zb = (z[1:] + z[:-1])/2
            zb = np.hstack([1.5*z[0] - z[1]/2, zb, 1.5*z[-1] - z[-2]/2])
        else:
            # This will be used when generating the interpolator,
            # if len(alpha_grid) == 1, it will be still fine to get a 3D data
            self.single_alpha = True
            zb = np.array([0])


        # Setup FWHM_gal, FWHM_tem file and calculate sig
        if type(self.FWHM_gal) == str:
            logging.info('Load the line-spread function %s from the folder ./instrument' % self.FWHM_gal)
            lsf_gal = np.genfromtxt(instrumentDir + self.FWHM_gal)
        else:
            lsf_gal = np.zeros([len(wave), 2])
            lsf_gal[:, 0] = wave
            lsf_gal[:, 1] = self.FWHM_gal
        if type(self.FWHM_tem) == str:
            logging.info('Load the line-spread function %s from the folder ./instrument' % self.FWHM_tem)
            lsf_tem = np.genfromtxt(instrumentDir + self.FWHM_tem)
        else:
            lsf_tem = np.zeros([len(wave), 2])
            lsf_tem[:, 0] = wave
            lsf_tem[:, 1] = self.FWHM_tem
        sig = utils.cal_degrade_sig(np.interp(wave, lsf_gal[:, 0], lsf_gal[:, 1], left=np.nan, right=np.nan),
                                    np.interp(wave, lsf_tem[:, 0], lsf_tem[:, 1], left=np.nan, right=np.nan),
                                    np.diff(wave)[0])

        if np.all(np.isnan(sig)) == True:
            raise ValueError("The outputs FWHM_gal is lower than the SSP FWHM_tem.")
        elif np.any(np.isnan(sig)) == True:
            logging.info("FWHM_gal is smaller than the SSP FWHM_tem in some wavelength pixels, will remove them and contiue the program...")
            mask_nansig = ~np.isnan(sig)
            wave = wave[mask_nansig]
            templates = templates[mask_nansig, :, :]
            sig = sig[mask_nansig]
        else:
            logging.info("FWHM_gal is larger than the SSP FWHM_tem for all wavelength pixels, continue the program...")


        self.templates = templates
        self.reg_dim = reg_dim
        self.wave = wave
        self.velscale = velscale
        self.age_grid = age_grid
        self.logage_grid = np.log10(age_grid) + 9
        self.metal_grid = metal_grid
        self.alpha_grid = alpha_grid
        self.age_grid_3d = age_grid_3d
        self.metal_grid_3d = metal_grid_3d
        self.alpha_grid_3d = alpha_grid_3d
        self.age_grid_2d = age_grid_3d[:, :, 0]
        self.metal_grid_2d = metal_grid_3d[:, :, 0]
        self.xb = xb
        self.yb = yb
        self.zb = zb
        self.lsf_gal = lsf_gal
        self.lsf_tem = lsf_tem
        self.sig = sig
        # self.templates_org = model.templates

        # Add the new wave
        if self.dlam == None:
            self.new_wave = self.wave
        else:
            self.new_wave = np.arange(wave[0], wave[-1], self.dlam)


        logging.info('======================SSP info======================')
        logging.info('SSP name:                 %s' % self.ssp_name)
        logging.info('library:                  %s' % self.library)
        logging.info('oversample factor:        %s' % self.factor)
        logging.info('FWHM_gal:                 %s' % self.FWHM_gal)
        logging.info('FWHM_tem:                 %s' % self.FWHM_tem)
        logging.info('wave interval (0.1nm):    %s' % self.dlam)
        logging.info('velscale (km/s):          %.2f' % velscale)
        logging.info('age_range (Gyr):          %s' % [float('{:.3f}'.format(i)) for i in age_grid[[0, -1]]])
        logging.info('metal_range (dex):        %s' % [float('{:.3f}'.format(i)) for i in metal_grid[[0, -1]]])
        logging.info('alpha_range (dex):        %s' % [float('{:.3f}'.format(i)) for i in alpha_grid[[0, -1]]])
        logging.info('single_alpha?:            %s' % self.single_alpha)
        logging.info('templates dim:            %s' % list(reg_dim))
        logging.info('n_templates:              %s' % np.prod(reg_dim))
        logging.info('SSP wave range:           %s' % [float('{:.3f}'.format(i)) for i in [self.wave[0], self.wave[-1]]])
        if self.wave_range==None:
            logging.info('Cube wave range:          %s' % [float('{:.3f}'.format(i)) for i in [self.new_wave[0], self.new_wave[-1]]])
        else:
            logging.info('Cube wave range:          %s' % [float('{:.3f}'.format(i)) for i in self.wave_range])
        logging.info('====================================================')



    def oversample(self):

        # Oversample the templates and integral the spectra again
        wave_oversampled = ndimage.interpolation.zoom(self.wave, self.factor, order=1)
        templatesOversampled = np.zeros([self.templates.shape[0]*self.factor] + list(self.templates.shape[1:]))


        # Setup FWHM_gal file and calculate sig
        sig_oversampled = utils.cal_degrade_sig(np.interp(wave_oversampled, self.lsf_gal[:, 0], self.lsf_gal[:, 1]),
                                                np.interp(wave_oversampled, self.lsf_tem[:, 0], self.lsf_tem[:, 1]),
                                                np.diff(wave_oversampled)[0])

        pool = Pool(processes=self.ncpu)
        results = []
        for i in range(templatesOversampled.shape[1]):
            for j in range(templatesOversampled.shape[2]):
                for k in range(templatesOversampled.shape[3]):
                    results.append(pool.apply_async(process_Oversampling_templates, (self.templates[:, i, j, k], self.factor, i, j, k, )))
        pool.close()
        pool.join()

        for result in results:
            starNew, i, j, k = result.get()
            templatesOversampled[:, i, j, k] = starNew

        self.templatesOversampled = templatesOversampled
        self.wave_oversampled = wave_oversampled
        self.sig_oversampled = sig_oversampled

        self.templatesTransposed = np.transpose(self.templatesOversampled, (1, 2, 3, 0))




    def load_ssp(self, templateDir):

        logging.info("Using the routine for '"+self.ssp_name+"_util.py'")
        spec = importlib.util.spec_from_file_location("", os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/ssp_utils/"+self.ssp_name+"_util.py")
        SSPLoaderModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(SSPLoaderModule)

        model = SSPLoaderModule.ssp(os.path.join(templateDir, self.library), FWHM_tem=self.FWHM_tem,
                                    age_range=self.age_range, metal_range=self.metal_range, alpha_range=self.alpha_range)

        return model


    def degrade_logrebin(self, velscale, lmin, lmax):

        lamRange_spmod = self.wave[[0, -1]]

        # Determine length of templates
        template_overhead = np.zeros(2)
        if lmin - lamRange_spmod[0] > 150.:
            template_overhead[0] = 150.
        else:
            template_overhead[0] = lmin - lamRange_spmod[0] - 5
        if lamRange_spmod[1] - lmax > 150.:
            template_overhead[1] = 150.
        else:
            template_overhead[1] = lamRange_spmod[1] - lmax - 5

        # Create new lamRange according to the provided LMIN and LMAX values, according to the module which calls
        constr = np.array([ lmin - template_overhead[0], lmax + template_overhead[1] ])
        idx_lam = np.where( np.logical_and(self.wave > constr[0], self.wave < constr[1] ) )[0]
        lamRange_spmod = np.array([ self.wave[idx_lam[0]], self.wave[idx_lam[-1]] ])

        star_eg= process_DegradingLogRebinning_templates(self.templatesOversampled[:, 0, 0, 0], self.wave, self.wave_oversampled, self.sig_oversampled, velscale, idx_lam, lamRange_spmod, 0, 0, 0)[0]
        templatesOversampledDegradedLogRebinned = np.zeros([star_eg.shape[0]] + list(self.templates.shape[1:]))

        pool = Pool(processes=self.ncpu)
        results = []
        for i in range(templatesOversampledDegradedLogRebinned.shape[1]):
            for j in range(templatesOversampledDegradedLogRebinned.shape[2]):
                for k in range(templatesOversampledDegradedLogRebinned.shape[3]):
                    results.append(pool.apply_async(process_DegradingLogRebinning_templates,
                                                    (self.templatesOversampled[:, i, j, k], self.wave, self.wave_oversampled, self.sig_oversampled, velscale, idx_lam, lamRange_spmod, i, j, k, )))
        pool.close()
        pool.join()

        for result in results:
            star, logLam, i, j, k = result.get()
            templatesOversampledDegradedLogRebinned[:, i, j, k] = star

        self.templatesOversampledDegradedLogRebinned = templatesOversampledDegradedLogRebinned / np.mean(templatesOversampledDegradedLogRebinned)
        self.logLam = logLam


def process_Oversampling_templates(star, factor, i, j, k):
    starNew = ndimage.interpolation.zoom(star, factor, order=3)  # Oversampling
    return starNew, i, j, k


def process_DegradingLogRebinning_templates(starNew, wave, wave_oversampled, sig, velscale, idx_lam, lamRange_spmod, i, j, k):
    starNew = utils.degrade_spec_ppxf(starNew, None, sig, gau_npix=None)[0]  # Degrading the oversampled spectra
    star = spectres(wave, wave_oversampled, starNew, fill=np.nan, verbose=False)  # The rebin is needed because the spectra is also rebinned
    star = star[idx_lam]
    starLogRebin, logLam, _ = utils.log_rebin(lamRange_spmod, star, velscale=velscale)
    return starLogRebin, logLam, i, j, k
