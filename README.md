# Code for "A Refined Model of Convectively-Driven Flicker in Kepler Light Curves"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4444282.svg)](https://doi.org/10.5281/zenodo.4444282)

This is the code implementing our model for stellar flicker in Kepler light curves. An arXiv link will be included here nearer to publication.

`analysis_and_plots.ipynb` contains most of our analysis and produces all of the plots in our paper.

`base.py` implements our core functions, which are used by `analysis_and_plots.ipynb`. This is likely the code most relevant to anyone wishing to build on our results. Note that this file includes functions implementing both our updated model and the prior version implemented by Cranmer et al. (2014).

`build_catalog.py` produces our merged catalog from a number of observational sources.

`merged_catalog.npy` contains our merged catalog, derived from multiple observational sources as described in our paper. See the comments in `build_catalog.py`, beginning at line 57, for information on column naming conventions and sources.

`hinode_data.ipynb` contains all of the code and analysis going into our use of Hinode/SOT data to determine the solar $\Theta$ (Appendix B of our paper).

`bandpass_correction/` contains the code and intermediary data for producing our Kepler bandpass correction factor (Section 3.6 and Appendix A of our paper).

`orig_data/` contains a few data files used by functions in `base.py` as well as a description of the other data files drawn from by `build_catalog.py` and where those files can be found. (Those files are included in our Zenodo repository but not this Git repository.)
