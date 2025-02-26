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
