# GHG-flux
quantifying greenhouse gas flux


## Installation

To install, clone the repository using e.g. `git clone ` , cd to the main directory and use  `pip install -e .` to install it as an editable python package.

Then run `pip install -r requirements.txt` to install dependencies.

## Usage
Processing- for drone flights  ingests a csv file with the following minimum columns:
- `timestamp` (datetime)
- `latitude` (float)
- `longitude` (float)
- `altitude_ato` (float) - altitude above takeoff
- `windspeed` (float)
- `winddir` (float)
- `temperature` (float)
- `pressure` (float)
It then requires a gas concentraiton column in ppm for each gas being measured. The column name should be the gas name, e.g. `co2`, `ch4`, `n2o`, etc. and the gases to be processed are defined in config.yaml. e.g.
- `ch4` (float) - in ppm
- `co2` (float) - in ppm
- `n2o` (float) - in ppm

Processing will then add columns for local easting and northing, and calculate a flux column for each gas using the wind speed and direction, and the gas concentration. This is calculated based on enhancements by normalising the gas flux to a baseline. The flux column is then interpolated to give an emissions flux.

## Drone Logs

The best way to convert encoded DJI logs is to use "djiparsetext", a C++ library.

Documentation can be found here: http://djilogs.live555.com/


