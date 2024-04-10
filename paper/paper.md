---
title: 'GasFlux: Algorithms for deriving gas emissions flux from mobile atmospheric measurements'
tags:
  - Python
  - atmospheric science
  - emissions quantification
  - mass balance
  - tracer inversion
authors:
  - name: Jamie V. McQuilkin
    orcid: 0009-0004-9872-5102
    corresponding: true
    affiliation: "1"
  - name: Grant Allen
    orcid: 0000-0002-7070-3620
    affiliation: "1"
affiliations:
 - name: University of Manchester, United Kingdom
 - date: 15 April 2024
bibliography: paper.bib
---

# Summary

GasFlux is a Python package developed to derive a mass emissions flux from  measurements of gas concentration and environmental variables such as windspeed, temperature and pressure. The primary use case is in converting mobile measurements (e.g. drone, aircraft, vehicle, satellite) of gas concentrations into an emissions estimate through various algorithmic strategies in concentration baselining and plume reconstruction.

The main use case for this is in measurements of greenhouse gas emissions (especially methane), with other applications in e.g. vulcanology and measurements of pollutants. The package arose in response to a lack of open source processing software for drone mass balances of industrial methane emissions, and it is intended to be as useful as possible for thiose working to understand and reduce pollutant emissions.

Many open source programs exist for specialised areas of eddy covariance, chamber flux and atmospheric inversion measurement techniques, reflecting the maturity and uptake of those methods. The development roadmap for this project therefore focuses on mobile measurements, typically from vehicles.

# Statement of need

To the author's knowledge, there is one publicly available package suitable for deriving emissions estimates from mobile measurements ([TERRA](https://github.com/ARQPDataTeam/TERRA/) - see @gordonDeterminingAirPollutant2015) which is implemented in the closed-source software Igor and intended for use with in-situ aircraft mass balances. Nothing exists as yet for tracer dispersion measurements or open-path column-integrated mass balances - these are both in the roadmap for `GasFlux`, although the initial release is limited to in-situ mass balance. Mobile measurements are the only reliable way of quantifying heterogeneous emissions at the spatial scale of 10s - 1000s of meters, which corresponds to most emissions of pollutants and greenhouse gases.

Currently a handful of companies offer site emissions measurement quantification using mobile measurements, mostly either using mass balance from satellites and aircraft (both autonomous and crewed) or tracer inversion. The algorithms used to process these are all closed-source, and thus far it has been difficult or impossible to review their quality or run independent analyses of raw data. Dozens of academic groups also conduct mass balance and tracer measurements, but these are all done by self-developed processing algorithms.

Algorithmic work in this area is published, but as a rule without detailed implementation code - see for example @allenMeasuringLandfillMethane2015, @allenDevelopmentTrialUnmanned2019, @shahNearFieldGaussianPlume2019, @yongLessonsLearnedUAV2024. This is a challenge for improvements to methods, and to the reproducibility of findings; `GasFlux` is designed in accordance with FAIR principles [[@wilkinsonFAIRGuidingPrinciples2016]]  and therefore records all input parameters and code versions, allowing truly reproducible analysis and comparisons.

# Availability and usage

`GasFlux` is available as a Python (>3.10) package via [PyPI](https://pypi.org/project/gasflux) and can be installed by the command `pip install gasflux`. Documentation is available on the [github repository](https://github.com/gasflux/gasflux/) or by invoking `gasflux --help`.

GasFLux offers the following features:

- Ease of use: given a high-quality input dataset, the default parameters are sufficient for most analyses
- Advanced baselining algorithms: `GasFlux` uses the `pybaselines` [package](https://github.com/derb12/pybaselines) [[@erbPybaselinesPythonLibrary2024]] to allow for compensation of drifting and shifting atmospheric baselines throughout measurements. Gas concentration enhancements above background are the basic unit of measurement, so this presents a significant improvement, with `pybaselines` collecting algorithms from many different fields of signal processing.
- Advanced interpolation algorithms: the initial implementation of `GasFlux` offers kriging interpolation via the `scikit-gstat` semivariogram modelling package [[@malickeSciKitGStatSciPyflavoredGeostatistical2022]]. This allows for fitting of autocorrelation models, the accuracy of which is an important component of interpolation.
- Designed for collaboration and in accordance to FAIR principles: `GasFlux` is licensed under AGPL-v3.0 and implemented in Python with modularity and testability in mind, (e.g. following a strategy design pattern for its analysis pipeline), and uses best practices in code style and continuous integration.

# Acknowledgements

This work was funded under a President's Doctoral Scholarship from the University of Manchester.

# References
