# *Welcome* *To* *HEAPY* 👋

### A unified toolkit for timing and spectral analysis of X-ray and gamma-ray transient data.

[![PyPI - Version](https://img.shields.io/pypi/v/heapy?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/heapy/)
[![License: GPL v3](https://img.shields.io/github/license/jyangch/heapy?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)


## Prerequisites

### `HEASoft`

Heapy will invoke certain software and commands from HEASoft, such as `xselect` and `ximage`. Please ensure that [`HEASoft`](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/#install) is correctly installed on your system, and that the [`Calibration Database`](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/install.html) (CALDB) for the mission (e.g., `Swift`)  you are processing is also properly installed.

### Fermi GBM Data Tools

Heapy obtains the orbital location and pointing information of `Fermi` by invoking `gbm_data_tools`. Therefore, if you require this functionality, please ensure that [`gbm_data_tools`](https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gdt-docs/index.html#) is correctly installed in advance.

### Fermi GBM Response Generator

Heapy generates the response matrix files for Fermi GBM by invoking [`gbm_drm_gen`](https://github.com/grburgess/gbm_drm_gen). It is recommended to install my forked Python packages, which have been fine-tuned to resolve compatibility issues with newer versions of `numpy` and `astropy`, and to use TTE data instead of CSPEC data. The specific installation procedure is as follows:
```bash
$ git clone https://github.com/jyangch/responsum.git
$ pip3 install ./responsum

$ git clone https://github.com/jyangch/gbmgeometry.git
$ pip3 install ./gbmgeometry

$ git clone https://github.com/jyangch/gbm_drm_gen.git
$ pip3 install ./gbm_drm_gen
```


## Installation

_heapy_ is available via `pip`:
```bash
$ pip install heapy
```


## Documentation

If you wish to learn about the usage, you may check the [`examples`](https://github.com/jyangch/heapy/tree/main/examples).


## License

_BaySpec_ is distributed under the terms of the [`GPL-3.0`](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
