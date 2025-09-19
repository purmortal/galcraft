1.5.0
=====

- Re-write scripts: SSPLoader.py; miles_util.py; pegase_interp_util.py; pegase_util.py; xshooter_util.py to make it possible to load multi-alpha templates directly, and reduce Removed many cumbersome procedures
- Refine configuration parameters, remove some that are only for testing

1.4.6
=====

- Revert some codes to v1.4.2 for SSPLoader.py to fix problems in loading other templates

1.4.5
=====

- Add function process_DegradingLogRebinning_templates and degrade_logrebin which is for estimating the true answer of GIST results

1.4.4
=====

- Add lsf_MUSE-alphaMC
- Change all the class name in SSP templates reader scripts to be "ssp"
- Refine the script name of each module

1.4.3
=====

- Limit numpy to be <2.0 and python<=3.12 given the current ebfpy issue

1.4.2
=====

- Add a version limit for numpy to be <=2.2.6 given numpy.fromstring is removed since then
- Fix the logging issue for Mac that is not printed in terminal (multiprocessing logging still not printing)

1.4.1
=====

- Update test kit files due to the change of FWHM strategy in `ssp_loader.py`
- Fix a problem in `ssp_loader.py`

1.4.0
=====

- Add script ``xshooter_util.py`` to load X-shooter SSP model
- Add X-shooter LSF file ``lsf_xshooter``
- Implement ``ssp_loader.py`` for X-shooter
- Update ``README.md``
- Fix the issue of `sig` in ``ssp_loader.py`` when `FWHM_gal` is smaller than `FWHM_tem` for some pixels, \change to just use the pixels with `FWHM_gal` > `FWHM_tem`.

1.3.1
=====

- Update ``README.md``

1.3.0
=====

- Create test kit for github actions
- Modify ``README.md``
- Refine ``MainProcess.py``
- add branch ``test_kit``

1.2.0
=====

- Clean codes
- Move SSP loading scripts into the new folder ``ssp_utils``
- Rename folder ``func`` to be ``modules``
- Optimize the codes in ``Mainprocess.py``

1.1.0
=====

- Remove unused codes and functions
- Update setup.py dependencies and version requirements
- Fix some logger integral format bugs (when ``python>=3.12``)
- Publish to ``pypi``

1.0.0
=====

- Build the package based on internal mwdatacube version v8.6.2
