<h1 align="center">HEAPY</h1>

<p align="center">
  <strong>A unified toolkit for timing and spectral analysis of X-ray and gamma-ray transient data.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/heapyx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/heapyx?color=4F46E5&labelColor=1f2937&logo=pypi&logoColor=white"></a>
  <a href="https://github.com/jyangch/heapy/tree/main/examples"><img alt="Examples" src="https://img.shields.io/badge/docs-examples-06B6D4?labelColor=1f2937"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img alt="License GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-9CA3AF?labelColor=1f2937"></a>
</p>

---

`HEAPY` is a Python toolkit for timing and spectral analysis of X-ray and
gamma-ray transient data. It pulls event and response data from mission
archives, reduces time-tagged-event (TTE) and image data into light curves
and OGIP spectra, separates burst signals from fitted backgrounds, and
measures durations, spectral lags, and source localizations — across
[`Fermi`](https://gammaray.nsstc.nasa.gov/gbm/)/GBM, GECAM, GRID,
[`Einstein Probe`](https://ep.bao.ac.cn/) (WXT/FXT), and
[`Swift`](https://swift.gsfc.nasa.gov/) (XRT/BAT). It bridges to
[`HEASoft`](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/) and
[`gbm_drm_gen`](https://github.com/grburgess/gbm_drm_gen) for mission-specific
reduction and response generation.


## Features

- **Data retrieval.** Pull event, position, and response files straight
  from mission archives by burst ID, UTC window, or observation ID
  (`gbmRetrieve`, `gecamRetrieve`, `gridRetrieve`, `epRetrieve`,
  `swiftRetrieve`).
- **Multi-mission event reduction.** One `Event` interface across
  Fermi/GBM TTE, GECAM, GRID, Einstein Probe (WXT/FXT), and Swift
  (XRT/BAT).
- **Light curves.** Extract and rebin light curves with automatic
  background fitting.
- **Signal detection.** Bayesian-block signal/background separation with
  three noise regimes — Poisson + polynomial background (`pgSignal`),
  Poisson + supplied background (`ppSignal`), and Gaussian (`ggSignal`).
- **OGIP spectra.** Generate source and background `PHA`/`PHA2` spectra,
  and — for Fermi/GBM — response matrices via `gbm_drm_gen`.
- **Timing analysis.** Duration measurements (`T90`/`Txx`) and
  spectral-lag estimation.
- **Localization & geometry.** Detector geometry, sky maps, and source
  localization.


## Installation

`HEAPY` is available on PyPI:

```bash
pip install heapyx
```

### Optional: `HEASoft`

For some missions (e.g., `Swift`), `HEAPY` invokes `HEASoft` tools such as
`xselect` and `ximage`. Ensure
[`HEASoft`](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/#install) is
installed, along with the
[`Calibration Database`](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/install.html)
(CALDB) for the mission you are processing.

### Optional: Fermi GBM response generator

To build response matrices for Fermi/GBM, `HEAPY` calls
[`gbm_drm_gen`](https://github.com/grburgess/gbm_drm_gen). The forked
packages below are fine-tuned for newer `numpy`/`astropy` and use TTE
instead of CSPEC data:

```bash
git clone https://github.com/jyangch/responsum.git
pip3 install ./responsum

git clone https://github.com/jyangch/gbmgeometry.git
pip3 install ./gbmgeometry

git clone https://github.com/jyangch/gbm_drm_gen.git
pip3 install ./gbm_drm_gen
```


## Documentation

Browse the [examples](https://github.com/jyangch/heapy/tree/main/examples)
for typical workflows end to end.


## License

`HEAPY` is distributed under the
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
