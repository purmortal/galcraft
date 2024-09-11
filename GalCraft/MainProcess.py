#!/usr/bin/env python

import os
import sys
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import ebf
import json
import optparse
from time import perf_counter as clock

from GalCraft.func.constant import *
import GalCraft.func.ssp_loader as ssp_loader
import GalCraft.func.binner as binner
import GalCraft.func.spec_generator as spec_generator
import GalCraft.func.cube_maker as cube_maker
import GalCraft.func.cot as cot
from GalCraft.func.log import Logger
from GalCraft._version import __version__


def run_GalCraft(CommandOption):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -   I N I T I A L I Z E   G A L C R A F T - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    t_init = clock()

    # - - - - - INITIALIZATION - - - - -

    # Initialize cube name and paths
    cube_name = CommandOption.configName
    setup_cube_name = 'setup_' + cube_name
    if os.path.isfile(CommandOption.defaultDir) == True:
        for line in open(CommandOption.defaultDir, "r"):
            if not line.startswith('#'):
                line = line.split('=')
                line = [x.strip() for x in line]
                if os.path.isdir(line[1]) == True:
                    if line[0] == 'configDir':
                        configDir = line[1]
                    elif line[0] == 'outputDir':
                        outputDir = line[1]
                    elif line[0] == 'modelDir':
                        modelDir = line[1]
                    elif line[0] == 'templateDir':
                        templateDir = line[1]
                else:
                    print("WARNING! "+line[1]+" specified as default "+line[0]+" is not a directory!")
    else:
        print("WARNING! "+CommandOption.defaultDir+" is not a file!")
    # Obtain ./instrument/ dir
    this_dir, this_filename = os.path.split(__file__)
    instrumentDir = os.path.join(this_dir, "instrument") + '/'
    # Imports the setup file
    with open(configDir + setup_cube_name + '.json', 'r') as f:
        params = json.load(f)
    # Setup the output folder
    filepath = outputDir + cube_name + '/'
    if os.path.exists(filepath)==False: os.mkdir(filepath[:-1])
    # For continue mode, change the logger mode from 'w' to 'a'
    if params['other_params']['mode'] == 'continue':
        params['other_params']['log_mode'] = 'a'
    logger = Logger(logfile=filepath + 'output.log', mode=params['other_params']['log_mode']).get_log()
    logger.info('Loaded the setup file (path %s)' % (configDir + setup_cube_name + '.json'))
    logger.info('Outputs DIR: %s' % filepath)
    # Setup number of cpu cores to use
    if params['other_params']['ncpu'] == None:
        n_cores = len(os.sched_getaffinity(0))
        logger.info('Change the CPU cores from %s to %s due to the CPU availability.' % (params['other_params']['ncpu'], n_cores) )
        params['other_params']['ncpu'] = n_cores
    else:
        logger.info('Run the framework using %s CPU cores.' % params['other_params']['ncpu'])


    # - - - - - INSTRUMENT SETUP - - - - -

    inst = params['cube_params']['instrument'].upper()
    # Here is to change the params "spatial_resolution" and "spatial_bin" to the assigned instrument
    # If "instrument"=="DIY", it will use the "spatial_resolution" and "spatial_bin" in the file. Basically
    # this means you are going to setup your own instrument
    # If "instrument"=="DEFAULT", it will generate a huge cube using all the particles
    if inst != 'DIY' and inst!='DEFAULT':
        with open(instrumentDir + inst + '.json', 'r') as f:
            inst_params = json.load(f)
        for key in inst_params:
            params['cube_params'][key] = inst_params[key]
    logger.info('Replace the instrument properties (spatial/spectral resolution/nbin) of %s with the ./instrument file.' % inst)


    # - - - - - LOAD TEMPLATES SPECTRA - - - - -

    logger.info('Loading SSP models...')
    t = clock()
    ssp_model = ssp_loader.model(templateDir, instrumentDir, params['ssp_params'], params['other_params'], logger)
    ssp_model.oversample()
    logger.info('SSP model has been successfully loaded, time elapsed: %.2f s' % (clock() - t))


    # - - - - - LOAD STELLAR CATALOG - - - - -

    logger.info('Loading the E-Galaxia model from %s' % (modelDir + params['other_params']['model_name']))
    t = clock()
    d_t = ebf.read(modelDir + params['other_params']['model_name'],'/')
    # Check the need to locate/rotate the E-Galaxia model.
    if 'vr' in d_t.keys():
        logger.info('Model has already been rotated, no location/rotation transformation is applied.')
        for key in params['oparams']:
            params['oparams'][key] = None
    else:
        logger.info('Specifing the location/rotation of the galaxy')
        cot.observe(d_t, params['oparams'])
    # Check extinction values
    if params['cube_params']['use_extinc'] == True:
        assert 'exbv' in d_t.keys(), "No extinction in the loaded model, process terminated."
    # Check the need to apply the flux calibration due to the total Galaxy Stellar mass.
    if params['other_params']['nparticles'] != None:
        nparticles = params['other_params']['nparticles']
    else:
        if 'shuffle' in params['other_params']['model_name']:
            original_model_name = '_'.join([x for x in params['other_params']['model_name'][:-4].split('_') if 'shuffle' not in x]) + '.ebf'
            d_original = ebf.read(modelDir + original_model_name,'/')
            nparticles = len(d_original['px'])
            del d_original
        else:
            nparticles = len(d_t['vr'])
    # In case nparticles changed due to the "debug" mode.
    if params['other_params']['mode'] == 'debug':
        nparticles = 500000
    logger.info('Number of particles = %s or Galaxy Stellar Mass' % nparticles)
    # Calculate mass_cali_factor for flux calibration
    if params['other_params']['gal_mass'] != None:
        mass_cali_factor = params['other_params']['gal_mass'] / nparticles
        logger.info('Input Galaxy Stellar mass is %s, then the calibration factor is %s.' % (params['other_params']['gal_mass'], mass_cali_factor))
    else:
        logger.info('No flux calibration due to Galaxy Stellar mass is applied.')
        mass_cali_factor = 1
    logger.info('E-Galaxia model has been successfully loaded, time elapsed: %.2f s' % (clock() - t))



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - -   A N A L Y S I S   M O D U L E S   - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    # - - - - - SPATIAL BINNING - - - - -

    if params['other_params']['mode'] != 'continue':
        logger.info('Start binning the model...')
        t = clock()
        d_t_l, statistic_count_l, x_edges_l, y_edges_l = binner.spatial_binner(d_t, params['cube_params'], params['other_params'],
                                                                               ssp_model.age_grid, ssp_model.metal_grid, ssp_model.alpha_grid,
                                                                               filepath, logger, configDir + setup_cube_name,
                                                                               params['oparams']['distance'], nparticles)
        logger.info('Binning process has been finished, time elapsed: %.2f s' % (clock() - t))
    else:
        logger.info('Continue the running from previous run...')
        t = clock()
        d_t_l, statistic_count_l, x_edges_l, y_edges_l = binner.spatial_binner_continue(params['cube_params'], params['other_params'],
                                                                                        filepath, logger, configDir + setup_cube_name)
        logger.info('Binning process has been finished, time elapsed: %.2f s' % (clock() - t))
    d_t = None


    # - - - - - START DATACUBE LOOP  - - - - -

    for cube_idx in range(len(statistic_count_l)):
        d_t = d_t_l[cube_idx]
        statistic_count = statistic_count_l[cube_idx]
        x_edges = x_edges_l[cube_idx]
        y_edges = y_edges_l[cube_idx]


    # - - - - - GENERATE DATACUBE  - - - - -

        if params['other_params']['mode'] == 'continue' and os.path.exists(filepath + 'data_cube_' + str(cube_idx) + '.fits'):
            logger.info("Data Cube No.%s exists in %s, will not be run in 'continue' mode" % (cube_idx+1, filepath))
            continue
        else:
            logger.info('Start generating the datacube No.%s...' % (cube_idx+1))
            t = clock()
            Spec_DataCube = spec_generator.Spec_Generator(d_t, x_edges, y_edges, statistic_count, ssp_model,
                                                          params['cube_params'], params['ssp_params'], params['other_params'],
                                                          filepath, logger, cube_idx)
            Spec_DataCube()
            data_cube = Spec_DataCube.data_cube
            data_cube = data_cube * mass_cali_factor
            logger.info('Calibrated the flux based on the input Galaxy Stellar Mass=%s.' % params['other_params']['gal_mass'])
            # Free memory
            Spec_DataCube = None
            d_t = None
            statistic_count = None
            logger.info('Stellar spectra have been generated, time elapsed: %.2f s' % (clock() - t))


    # - - - - - WRITE DATACUBE  - - - - -

        cube_maker.write_cube(data_cube, params, x_edges, y_edges, ssp_model.new_wave, filepath, logger, cube_idx, ssp_model.velscale, __version__)
        logger.info('Data Cube No.%s has been written in %s, total time elapsed: %.2f s' % (cube_idx+1, filepath , clock() - t))




    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  F I N A L I S E   T H E   A N A L Y S I S  - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    # - - - - - SAVE CONFIG  - - - - -

    logger.info('Write the json file into the output folder.')
    with open(filepath + 'setup.json', 'w') as f:
        json.dump(params, f, indent=2)
    logger.info('All the datacubes have been generated successfully, total time elapsed: %.2f s' % (clock() - t_init))
    logger.info("GalCraft completed successfully.")




# ============================================================================ #
#                           M A I N   F U N C T I O N                          #
# ============================================================================ #
def main(args=None):

    # Capture command-line arguments
    parser = optparse.OptionParser(usage="%GalCraft [options] arg")
    parser.add_option("--config",      dest="configName", type="string", \
            help="State the name of the MasterConfig file.")
    parser.add_option("--default-dir", dest="defaultDir", type="string", \
            help="File defining default directories for input, output, configuration files, and spectral templates.")
    (CommandOption, args) = parser.parse_args()

    # Check if required command-line argument is given
    assert CommandOption.configName != None, "Please specify the path of the Configuration.json file to be used. Exit!"

    # Run the framework
    run_GalCraft(CommandOption)


if __name__ == '__main__':
    main()
