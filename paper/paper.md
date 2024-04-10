---
title: 'GasFlux: Algorithms for deriving gas emissions flux using mass balance and tracer methods
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
    affiliation: 1
  - name: Grant Allen
    orcid: 0000-0002-7070-3620
    affiliation: 1
affiliations:
 - name: Department of Earth and Environmental Sciences, University of Manchester, United Kingdom
   index: 1
date: 15 April 2024
bibliography: paper.bib
---

## Summary

`GasFlux` is a Python package developed to derive a mass emissions flux from  simultaneous geolocated measurements of gas concentration and environmental variables such as wind speed, temperature and pressure. These are primarily taken from mobile platforms such as drones, aircraft, ground vehicles, or satellites and processed using a range of algorithmic strategies to perform background concentration normalisation, plume reconstruction and other related functions.

The main application for this is in measurements of greenhouse gas emissions (especially methane), with other applications in e.g. volcanology and measurements of pollutants from localised sources. The package was developed in response to a lack of open source processing software for mass balance measurements of industrial methane emissions using drones (also known as UAVs, RPAS, or UAS), and it is intended to be as useful as possible for those working to understand, quantify and reduce pollutant emissions.

A large amount of specialised open-source software exists for fixed methods of measurement - e.g. eddy covariance, chamber flux and atmospheric inversion  techniques - reflecting the maturity and uptake of those methods. The development roadmap for this software therefore focuses on measurements where there is a dynamic element, where the sensor or its focused is moved through a plume.

## Statement of need

Mobile measurements are the only reliable way of quantifying heterogeneous emissions at the spatial scale of tens to thousands of meters, which corresponds to most emissions of pollutants and greenhouse gases. To the authors' knowledge, there is one publicly available package suitable for deriving emissions estimates from mobile measurements ([TERRA](https://github.com/ARQPDataTeam) - see @gordonDeterminingAirPollutant2015, and further developments by Environment and Climate Change Canada), aimed at mass balances using crewed aircraft and implemented in Igor Pro (WaveMetrics, Inc., Lake Oswego, OR, USA). The initial release of `GasFlux` is likewise limited to in-situ mass balance emissions calculation, but column-integrated and tracer inversion algorithms are planned for future releases.

Currently a handful of companies offer site emissions measurement quantification using mobile measurements, mostly either using mass balance from satellites and aircraft (both autonomous and crewed) or tracer inversion. The algorithms used to process these are typically closed-source, making it difficult to fully review their quality or run independent analyses of raw data. Many academic groups also conduct mass balance and tracer measurements, but these are usually bespoke and often not shared in a systematic way with the wider community, with `TERRA` being the primary exception to this.

Algorithmic work in this area is published, but as a rule without detailed implementation code - see for example @allenMeasuringLandfillMethane2015, @allenDevelopmentTrialUnmanned2019, @shahNearFieldGaussianPlume2019, @yongLessonsLearnedUAV2024. This is a challenge for improvements to methods, and to the reproducibility of findings; `GasFlux` is designed in accordance with FAIR principles [[@wilkinsonFAIRGuidingPrinciples2016]]  and therefore records all input parameters and code versions, allowing truly reproducible analysis and comparisons.

## Availability and usage

`GasFlux` is available as a Python (>=3.10) package via [PyPI](https://pypi.org/project/gasflux) and can be installed by the command `pip install gasflux`. Documentation is available on the [github repository](https://github.com/gasflux/gasflux/) or by invoking `gasflux --help`.

GasFlux offers the following features:

- Ease of use: given a high-quality input dataset, the default parameters are sufficient for most analyses
- Advanced baselining algorithms: `GasFlux` uses the `pybaselines` [package](https://github.com/derb12/pybaselines) [[@erbPybaselinesPythonLibrary2024]] to allow for compensation of drifting and shifting atmospheric baselines throughout measurements. Gas concentration enhancements above background are the basic unit of measurement, so this presents a significant improvement, with `pybaselines` collecting algorithms from many different fields of signal processing.
- Advanced interpolation algorithms: the initial implementation of `GasFlux` offers kriging interpolation via the `scikit-gstat` semivariogram modelling package [[@malickeSciKitGStatSciPyflavoredGeostatistical2022]]. This allows for fitting of autocorrelation models, the accuracy of which is an important component of interpolation.
- Designed for collaboration and in accordance to FAIR principles: `GasFlux` is licensed under AGPL-v3.0 and implemented in Python with modularity and testability in mind, (e.g. following a strategy design pattern for its analysis pipeline), and aims to use best practices in code style and continuous integration.

## Acknowledgements

Jamie McQuilkin and Grant Allen were funded under a President's Doctoral Scholarship from the University of Manchester; Grant Allen was part-funded by the Natural Environment Research Council funding - NE/X014649/1 - "Mobile Observations and quantification of Methane Emissions to inform National Targeting, Upscaling and Mitigation" project.

## References
