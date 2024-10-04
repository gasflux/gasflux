
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Tests](https://github.com/gasflux/gasflux/workflows/CI/badge.svg)](https://github.com/gasflux/gasflux/actions?query=workflow%3A%22CI%22)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![format - Black](https://img.shields.io/badge/format-black-000000.svg)](https://github.com/psf/black)
[![types - mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://github.com/python/mypy)

# GasFlux

`pip install gasflux` \
`gasflux --help`

GasFlux is a tool for processing atmospheric gas concentration data and windspeeds into mass emissions fluxes, with principle applications to greenhouse gas measurement and vulcanology. Currently it works with in situ ("sniffing") data from UAVs and other aircraft, using mass balance as a paradigm and kriging as an interpolation strategy, but the intention is to expand this to other kinds of sampling and processing strategies, such as open-path and tracer methods.

It is released under the AGPLv3 license as a free and open-source project - comments, pull requests, issues and co-development are warmly welcomed. Currently development is co-ordinated by Jamie McQuilkin ([@pipari](https://github.com/pipari)) at the UAV Greenhouse Gas group at the University of Manchester.

## User Installation

The package is available on PyPi and can be installed using `pip install gasflux`.

## Usage

The package interface is in active development. Currently it ingests a data csv file (or folder containing only data csv files) and a config file that dictates the parameters of the analysis.

This is done through the syntax `gasflux process <input_file> --config <config_file>`.

### The config file

The default gasflux_config.yaml is located in the package source. It can be generated in a supplied directory using `gasflux generate-config <path>`. If a directory is supplied to `gasflux process` and a config is not also explicitly supplied, the package will look for one config file in that directory or its subdirectories and attempt to process all csv files in that directory and subdirectories. If multiple or no config is found, it will raise an error. If supplying a single csv file for processing (rather than a directory), only the parent directory will be searched for config files (not its subdirectories).

Through it, variables can be passed to the [scikit-gstat](https://scikit-gstat.readthedocs.io/en/latest/) package used for kriging and the [pybaselines](https://pybaselines.readthedocs.io/en/latest/) package used for background correction.

### The data file

Input data files must be csv-type (i.e. readable by `pandas`) and have the following columns (all lower case):

- `timestamp` (datetime)
- `latitude` (float)
- `longitude` (float)
- `height_ato` (float) - height above takeoff
- `windspeed` (float) - in m/s, as measured or inferred at each measurement point
- `winddir` (float) - in standard 0-360 degree format, relative to the earth
- `temperature` (float) in degrees celsius
- `pressure` (float) - in hPa/mBar

At least one gas concentration in ppm is also required. The column name should be the gas name, e.g. `co2`, `ch4`, `n2o`, etc.

The gas should be entered in the gasflux_config.yaml file along with a range of concentrations in ppmv, e.g.:

```
gases:
  ch4: [1.5, 500]
  co2: [300, 5000]
  c2h6: [-0.5, 10]
```

Ensuring input data are sufficient and correctly formatted is non-trivial and important, but is left to the user. Data sources vary enormously so it is difficult to generalise this part of the analysis - in many cases these will be a mix of flight logs, GPS, one or more anemometers, one or more gas sensors, a thermometer, hygrometer, barometer.

Synchronisation and fusion of these data sources is important and should be given great attention - there are several ways to do this, including GPS logging on each sensor, recording everything on a single device, or NTP server synchronisation. Care should also be taken to avoid loss of data through resampling or interpolation.

One way to convert encoded DJI logs is to use `djiparsetext`, a C++ library available on github [here](https://github.com/uav4geo/djiparsetxt) and documented [here](http://djilogs.live555.com/).

## Development

### Installation

To install, clone the repository using e.g. `git clone` and use  `pip install -e .` to install it as an editable python package.

It's highly recommended to use a virtual environment to manage dependencies. If you're not currently using one, [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) is a good option.

Then run `pip install -r requirements.txt` and `pip install -r dev-requirements.txt` to install the required dependencies.

User requirements.txt is generated using `pigar generate` rather than `pip freeze`.
