# Code for "A Refined Model of Convectively-Driven Flicker in Kepler Light Curves"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4444282.svg)](https://doi.org/10.5281/zenodo.4444282)

This is the code implementing our model for stellar flicker in Kepler light curves.

`base.py` implements our core functions

`build_catalog.py` produces our merged catalog from a number of observational sources

`analysis_and_plots.ipynb` contains most of our analysis and produces all of the plots in our paper

`hinode_data.ipynb` contains all of the code and analysis going into our use of Hinode/SOT data to determine the solar $\Theta$.

`bandpass_correction/` condains the code and intermediary data for producign our Kepler bandpass correction factor