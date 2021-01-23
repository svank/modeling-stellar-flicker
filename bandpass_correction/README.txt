This is Steve's code for producing our Kepler-bandpass correction factor (as in
Section 3.6 and Appendix B of our paper). Specifically, there are two IDL *.pro
for conducting the analysis, a number of *.out files containing the curated
output of step1, and step2_output.txt containing the collected output of step2.

Not included in our code repository is two input files:

kepler_response_hires1.txt: 
	Kepler Instrument Response Function (high resolution)
	Source: Kepler Instrument Handbook
		(http://keplergo.arc.nasa.gov/Instrumentation.shtml)
	Author: Jeff Van Cleve (NASA ARC)
	version: 1.0
	date: 2009/11/05

lejeune_lcbp00.cor: A re-named copy of lcpb00.cor, one of the data files from:
	"A standard stellar library for evolutionary synthesis. I. Calibration of
	theoretical spectra."
	Lejeune T., Cuisinier F., Buser R.
	<Astron. Astrophys. Suppl. Ser. 125, 229 (1997)>
	=1997A&AS..125..229L
	
	This file contains corrected flux distributions for [M/H] =  0.0
