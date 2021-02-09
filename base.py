#!/usr/bin/env python3

import astropy.constants as consts
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from numba import njit
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats

import functools

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

M_sun = 1 # Solar masses
M_sun_grams = 1.998e33
T_sun = 5770 # K
logg_sun = 4.438 # log-cgs units
ν_sun = 3.106e-3 # Hz
Ma_sun = 0.26   # As given in Cranmer 2014
                # Used in obsoleted scaling relations, and theta <-> phi
Z_sun = 0.01696 # From Grevesse & Sauval (1998)
R_sun_cm = 6.957e10
k_B = consts.k_B.cgs.value
m_h = consts.m_p.cgs.value
G = consts.G.cgs.value
sigma_sb = consts.sigma_sb.cgs.value

# n.b. this is no longer in use anywhere by default
DEFAULT_BETA = 8.7

# Note: see comments in build_catalog.py for documentation of the catalog format



# ---- General-purpose functions ----



def load_catalog(fname='merged_catalog.npy', raw=False):
    cat = np.load(fname)
    if not raw:
        cat = cat[cat['has_H']]
    return cat
        

def _F8_from_logg(logg):
    """Calculates F8 flicker values from log(g)
    Bastien 2016's Eqn 4.
    
    log g = 1.3724221 - 3.5002686 x - 1.6838185 x^2 - 0.37909094 x^3
    x = log10(F8)
    """
    # ax^3 + bx^2 + cx + d = 0
    a = -0.37909094
    b = -1.6838185
    c = -3.5002686
    d =  1.3724221 - logg
    
    # Numpy solves for roots generally, computing "the eigenvalues of
    # the companion matrix"
    roots = np.roots([a, b, c, d])
    # Select real root
    root = roots[np.imag(roots) == 0]
    root = np.real(root)
    
    if root.size != 1:
        raise RuntimeError("Cubic root is ambiguous")
    
    return 10**root

F8_from_logg = np.vectorize(_F8_from_logg)

def F8_to_logg(F8):
    """Calculates logg from an F8 value using
    Bastien 2016's Eqn 4.
    
    log g = 1.3724221 - 3.5002686 x - 1.6838185 x^2 - 0.37909094 x^3
    x = log10(F8)
    """
    a = -0.37909094
    b = -1.6838185
    c = -3.5002686
    d =  1.3724221
    x = np.log10(F8)
    return a * x**3 + b * x**2 + c * x + d



# ---- Functions implementing the "old" (Cranmer+2014) model ----



def calc_σ_old(Teff, M, logg, S=1, Φ=None, override_exponent=1.1):
    """Calculates RMS amplitude σ of photospheric continuum intensity variations
    Cranmer 2014's Eqn 1
    """
    ν_max = calc_ν_max(logg, Teff)
    if Φ is None:
        Ma = calc_Ma_old(logg, Teff)
        Ma *= S
        
        Φ = calc_Φ_old(Ma)
        
        # Clamp Φ at zero
        Φ = 0 * (Φ < 0) + Φ * (Φ >= 0)
    
    # Note Cranmer (2014) used an exponent of 1.03 (see Samadi Figure B.1)
    # Here we use the 1.10 from the main text of Samadi (2013b)
    return 0.039 * ( (Teff / T_sun) ** (3/4)
                     * (M_sun * ν_sun / M / ν_max) ** (1/2)
                     * Φ**2 )**override_exponent

def calc_σ_cranmer_2014(Teff, M, logg, S=1, Φ=None):
    return calc_σ_old(Teff, M, logg, S, Φ, override_exponent=1.03)

def calc_ν_max(logg, Teff):
    """Calculates peak p-mode frequency
    Cranmer 2014's Eqn 2
    """
    return ν_sun * 10**logg / 10**logg_sun * (T_sun / Teff)**0.5

def calc_Ma_old(logg, Teff):
    """Calculates Mach number
    Cranmer 2014's Eqn 3
    """
    return 0.26 * (Teff / T_sun)**2.35 * (10**logg_sun / 10**logg) ** 0.152

def calc_Φ_old(Ma, make_monotonic=True):
    """Calculates the temperature fluctuation amplitude
    Cranmer 2014's Eqn 4
    """
    A = -.59/.26/.26 # = -8.73
    B = 2.3/.26      # =  8.85
    C = -0.67        # = -0.67
    
    max_val = ( (-B**2 / (4*A)) + C )
    quadratic_val = A * Ma**2 + B * Ma + C
    
    # If Ma is beyond the Ma of the maximal Φ,
    # clamp to the maximal Φ instead
    return ((make_monotonic * Ma > (-B/2/A)) * max_val
            + ((not make_monotonic) or Ma <= (-B/2/A)) * quadratic_val)

def calc_phi_old(*args, **kwargs):
    """Turns out putting unicode in your function names makes them hard to type
    """
    return calc_Φ_old(*args, **kwargs)

def calc_F8_from_σ_old(logg, Teff, σ, S=1):
    """Cranmer 2014's Eqn 8"""
    ν_8 = 1 / (8 * 3600)
    ν_max = calc_ν_max(logg, Teff)
    Ma = calc_Ma_old(logg, Teff) * S
    
    τ_eff = 300 * (ν_sun * Ma_sun / ν_max / Ma)**0.98
    
    return σ * np.sqrt(1 - 2 / np.pi * np.arctan(4 * τ_eff * ν_8))

def calc_F8_old(logg, Teff, M, S=1, Φ=None):
    σ = calc_σ_old(Teff, M, logg, S, Φ)
    return calc_F8_from_σ_old(logg, Teff, σ, S)

def calc_cranmer_S(Teff):
    value = 1 / (1 + (Teff - 5400) / 1500)
    return 1 * (Teff <= 5400) + value * (Teff > 5400)

def fit_S(logg_arr, Teff_arr, M_arr, F8obs_arr, max_S=2, too_large_fill=0):
    S = np.zeros_like(logg_arr)
    
    for i in range(len(S)):
        logg = logg_arr[i]
        Teff = Teff_arr[i]
        M = M_arr[i]
        F8obs = F8obs_arr[i]
        
        # The objective function to be minimized
        obj = lambda S: calc_F8_old(logg, Teff, M, S) - F8obs
        
        try:
            S[i] = scipy.optimize.newton(obj, x0=calc_cranmer_S(Teff))
        except:
            S[i] = float('nan')
            #s0 = calc_cranmer_S(Teff)
            #print(i, s0, calc_F8_old(logg, Teff, M, s0), calc_F8_old(logg, Teff, M, 1), F8obs)
    
    # Set values >= max to fill value
    S[S > max_S] = too_large_fill
    return S

def fit_Φ(logg_arr, Teff_arr, M_arr, F8obs_arr):
    Φ = np.zeros_like(logg_arr)
    
    for i in range(len(Φ)):
        logg = logg_arr[i]
        Teff = Teff_arr[i]
        M = M_arr[i]
        F8obs = F8obs_arr[i]
        
        # The objective function to be minimized
        obj = lambda Φ: calc_F8_old(logg, Teff, M, Φ=Φ) - F8obs
        
        try:
            Φ[i] = scipy.optimize.newton(obj, x0=1)
        except:
            Φ[i] = float('nan')
    
    return Φ

def calc_N_gran(R, Teff, logg):
    """Note this uses the old Lambda, not our updated scaling"""
    Lambda = (Teff / T_sun) / (10**logg / 10**logg_sun) * 1e8 # gran size, cm
    return 2 * np.pi * R**2 / Lambda**2




# ---- And now the code implementing our new model! ----
# In general, each function implements one equation or one small step from our
# model. Note that the use of the beta parameter (the free parameter
# connecting granular sizes to scale heights) has been excised from the final
# version of our model. But beta still appears throughout these functions
# (with any given value ignored) (a) for some backwards compatability with my
# old notebooks, and (b) to facilitate quick comparisons of model predictions
# with the old (beta-containing) and new (beta-free) granular size expressions
# by swapping in the beta-containing Lambda function.




def calc_phi_new(Ma):
    # Phi is just Theta / Theta_sun
    return calc_theta(Ma) / calc_theta(Ma_sun)

@njit
def calc_theta(Ma):
    # As determined by our fit
    A = 20.98
    B = 3.5e6
    e2 = -0.84
    s = 5.29
    return 1 / (A * Ma ** (-2*s) + B * Ma ** (e2)) ** (1/s)

@njit
def calc_theta_min(Ma):
    """Calculates the minimum possible Theta according to the envelope"""
    # As determined by our fit
    return 0.62 * calc_theta(Ma)
   
@njit
def calc_theta_max(Ma):
    """Calculates the maximum possible Theta according to the envelope"""
    # As determined by our fit
    return 1.27 * calc_theta(Ma)

@njit
def calc_theta_ratio_to_envelope(theta_emp, Ma):
    # Takes in empirical Thetas---that is, a Theta backed out of the model given
    # an observational F8. Returns 1 if that empirical Theta is within the range
    # of the Theta values given by our envelope fit, given a model-predicted
    # Mach number. Otherwise returns the ratio of the empirical Theta to the
    # nearest edge of the envelope range.
    bound_lower = calc_theta_min(Ma)
    bound_upper = calc_theta_max(Ma)
    ratio_lower = theta_emp / bound_lower
    ratio_upper = theta_emp / bound_upper
    out = np.ones_like(ratio_lower)
    out[theta_emp < bound_lower] = ratio_lower[theta_emp < bound_lower]
    out[theta_emp > bound_upper] = ratio_upper[theta_emp > bound_upper]
    return out

def _calc_Ma_from_theta(theta, theta_fcn=calc_theta):
    """Numerically inverts the Theta-Ma relation and finds
    a Ma for a given Theta"""
    if np.isnan(theta):
        return np.nan
    f = lambda Ma: np.abs(theta_fcn(Ma) - theta)
    # Bounds are needed, in part, to prevent guesses of 0
    return scipy.optimize.minimize_scalar(f, method="Bounded", bounds=(0.01, 2)).x

calc_Ma_from_theta = np.vectorize(_calc_Ma_from_theta)

aesopus_data = None

def load_aesopus():
    # Loads the Aesopus data and injects it into the global namespace.
    # The data is a bit heavy, so having this code function-ified allows it to be
    # disabled if desired.
    global aesopus_R_vals, aesopus_data
    global aesopus_T_idx_mapper, aesopus_Z_idx_mapper
    from pathlib import Path
    
    # Load the data
    files = list(Path("orig_data/aesopus/").glob("Z*"))
    aesopus_data = np.zeros((len(files), 67, 91))
    aesopus_Z_vals = np.zeros(len(files))
    aesopus_R_vals = 10**np.arange(-8, 1.01, .1)
    for i, f in enumerate(sorted(files)):
        aesopus_Z_vals[i] = str(f).split("Z")[1]
        d = np.genfromtxt(f, skip_header=329)
        aesopus_T_vals = 10**d[:, 0]
        aesopus_data[i] = d[:, 1:] 
    
    # Prep interpolation functions
    import scipy.interpolate
    aesopus_T_idx_mapper = scipy.interpolate.interp1d(aesopus_T_vals, np.arange(aesopus_T_vals.size), kind='linear')
    aesopus_Z_idx_mapper = scipy.interpolate.interp1d(aesopus_Z_vals, np.arange(aesopus_Z_vals.size), kind='linear') 
    
    # There are many duplicate T and Z values in our catalog,
    # so caching lookups gives a nice speed boost
    aesopus_T_idx_mapper = functools.lru_cache(maxsize=5000)(aesopus_T_idx_mapper)
    aesopus_Z_idx_mapper = functools.lru_cache(maxsize=5000)(aesopus_Z_idx_mapper)

load_aesopus()

@njit
def _aesopus_interpolate(z1, z2, t1, t2, z1r, z2r, t1r, t2r):
    """ This is a line of code pulled from _find_rho so that
        it can be cleanly jit-ed. """
    return ( 10**aesopus_data[z1, t1] * z1r * t1r
           + 10**aesopus_data[z1, t2] * z1r * t2r
           + 10**aesopus_data[z2, t1] * z2r * t1r
           + 10**aesopus_data[z2, t2] * z2r * t2r )

def _find_rho(T_val, Z_val, logg_val, and_kappa=False):
    if np.isnan(T_val) or np.isnan(Z_val) or np.isnan(logg_val):
        if and_kappa:
            return np.nan, np.nan
        return np.nan
    # Find the indices of the two nearest T values
    T_idx = aesopus_T_idx_mapper(T_val)
    t1, t2 = int(np.floor(T_idx)), int(np.ceil(T_idx))
    # Find the distance (in index-space) to each of those nearest T values
    t1r = 1 - (T_idx - t1)
    t2r = 1 - t1r
    
    # Likewise for Z
    Z_idx = aesopus_Z_idx_mapper(Z_val)
    z1, z2 = int(np.floor(Z_idx)), int(np.ceil(Z_idx))
    z1r = 1 - (Z_idx - z1)
    z2r = 1 - z1r
    
    κ = _aesopus_interpolate(z1, z2, t1, t2, z1r, z2r, t1r, t2r)
    ρ = aesopus_R_vals * (1e-6 * T_val)**3
    
    μ = 7/4 + .5 * np.tanh( (3500-T_val) / 600)
    H = k_B * T_val / μ / m_h / 10**logg_val
    
    τ = κ * ρ * H
    
    # This version doesn't extrapolate
    #ρ_idx = np.interp(2/3, τ, np.arange(τ.size))
    ρ_val = np.interp(2/3, τ, ρ, left=np.nan, right=np.nan)
    
    # This version extrapolates
    #ρ_idx = scipy.interpolate.interp1d(τ, np.arange(τ.size), kind='linear', fill_value="extrapolate")(2/3)
    #ρ_val = scipy.interpolate.interp1d(τ, ρ, kind='linear', fill_value="extrapolate")(2/3)
    
    if and_kappa:
        #κ_idx = scipy.interpolate.interp1d(τ, np.arange(κ.size), kind='linear', fill_value="extrapolate")(2/3)
        #κ_val = scipy.interpolate.interp1d(τ, κ, kind='linear', fill_value="extrapolate")(2/3)
        
        #κ_idx = np.interp(2/3, τ, np.arange(κ.size))
        κ_val = np.interp(2/3, τ, κ, left=np.nan, right=np.nan)
         
        return ρ_val, κ_val
    return ρ_val

find_rho = np.vectorize(_find_rho)
rho_sun = find_rho(T_sun, Z_sun, logg_sun) 

def FeH_to_Z(FeH):
    return Z_sun * 10**FeH

def Z_to_FeH(Z):
    return np.log10(Z / Z_sun)

def calc_convective_turnover_time(Teff):
    return 0.002 + 314.24 * np.exp(-(Teff/1952.5) - (Teff/6250)**18)

def calc_bandpass_correction(Teff, logg):
    """Returns a factor that converts bolometric flicker to Kepler flicker.
    
    That is, this quantity should multiply model predictions or divide observed
    F8 to put prediction and observation on the same footing"""
    T_src, logg_src, sig_mult_src = np.genfromtxt("orig_data/bandpass_adjustment.txt", skip_header=4, unpack=True)
    sig_mult = scipy.interpolate.griddata((T_src, logg_src), sig_mult_src, (Teff, logg), 'linear')
    return sig_mult

mu = lambda Teff: 7/4 + .5 * np.tanh((3500-Teff)/600)

c_s = lambda Teff: np.sqrt((5/3) * k_B * Teff / mu(Teff) / m_h)

def calc_Ma(logg, Teff, Z):
    C = 6.086e-4 * Teff**1.406 * (10**logg)**-0.157 * (10**Z_to_FeH(Z))**0.098
    rho = find_rho(Teff, Z, logg)
    v = (sigma_sb * Teff**4 / (C * rho)) ** (1/3)
    Ma = v / c_s(Teff)
    return Ma

def H_p(T, logg):
    return k_B / mu(T) / m_h * T / 10**logg

def Lambda_beta(T, logg, beta=DEFAULT_BETA):
    return beta * H_p(T, logg)

def Lambda_Tram(T, logg, beta=None):
    """Calculates granule diameter Lambda via Trampedach+2013 eqn 19.
    
    The beta parameter is kept for drop-in compatability with Lambda(),
    but is not used.
    """
    # Note: A_gran is a *diameter*, not an area.
    log_A_gran = 1.3210 * np.log10(T) - 1.0970 * logg + 0.0306
    A_gran_Mm = 10**log_A_gran
    # To cm
    return 100 * 1e6 * A_gran_Mm

# n.b. For a quick hack to use the beta-based Lambda in calculations, just do
# `base.Lambda = base.Lambda_beta`. To undo, use `base.Lambda = base.Lambda_Tram`
Lambda = Lambda_Tram
    
def tau_g(T, logg, Z, beta=DEFAULT_BETA):
    rho, kappa = find_rho(T, Z, logg, and_kappa=True) 
    return Lambda(T, logg, beta) * rho * kappa

def tau_eff(Teff, logg, Z, Ma=None, beta=DEFAULT_BETA):
    if Ma is None:
        Ma = calc_Ma(logg, Teff, Z)
    
    w_rms = Ma * c_s(Teff)
    return Lambda(Teff, logg, beta) / w_rms

def calc_sigma(T, M, logg, Z, beta=DEFAULT_BETA, Ma=None, phi=None):
    if Ma is None:
        Ma = calc_Ma(logg, T, Z)
    if phi is None:
        theta = calc_theta(Ma)
    else:
        theta = phi * calc_theta(Ma_sun)
    M = M * M_sun_grams
    prefix = 12 / np.sqrt(2) / np.sqrt(2*np.pi*G)
    sigma = prefix * np.sqrt(tau_g(T, logg, Z, beta)) * Lambda(T, logg, beta) * np.sqrt(10**logg) / np.sqrt(M) * theta ** 2
    # Return in units of ppt
    return sigma * 1000

def calc_F8_from_sigma(logg, Teff, Z, σ, Ma=None, beta=DEFAULT_BETA):
    """Implements Equation 12 of our paper, except for the bandpass correction"""
    ν_8 = 1 / (8 * 3600)
    τ_eff = tau_eff(Teff, logg, Z, Ma, beta)
    
    return σ * np.sqrt(1 - 2 / np.pi * np.arctan(4 * τ_eff * ν_8))

def calc_σ_from_F8(logg, Teff, Z, F8, Ma=None, beta=DEFAULT_BETA):
    """Implements Equation 12 of our paper, except for the bandpass correction"""
    ν_8 = 1 / (8 * 3600)
    τ_eff = tau_eff(Teff, logg, Z, Ma, beta)
    
    return F8 / np.sqrt(1 - 2 / np.pi * np.arctan(4 * τ_eff * ν_8))

def calc_F8(logg, T, M, Z, phi=None, Ma=None, bp_cor=True, beta=DEFAULT_BETA):
    """Produce a model-predicted F8.
    
    Requires stellar parameters log_g, Teff, mass and heavy-element mass
    fraction Z.
    
    Passed values of phi (Theta/Theta_sun) and Mach number Ma can be provided
    to override those that would otherwise be used. (Computing Ma can be a
    heavy operation, so pre-computing Ma values before multiple F8 calculations
    can be beneficial.)
    
    If bp_cor==True (the default), the Kepler bandpass correction factor is
    applied, meaning the output of the function is a Kepler-like F8 prediction.
    When bp_cor==False, the function output is a bolometric F8.
    
    The beta argument is not used unless the beta-based Lambda function is
    enabled (see comment after the definition of Lambda_Tram).
    """
    
    sigma = calc_sigma(T, M, logg, Z, phi=phi, Ma=Ma, beta=beta)
    F8 = calc_F8_from_sigma(logg, T, Z, sigma, Ma=Ma, beta=beta)
    if bp_cor:
        F8 *= calc_bandpass_correction(T, logg)
    return F8

def calc_theta_from_F8(F8, logg, Teff, M, Z, beta=DEFAULT_BETA):
    """Backs out a Theta value from a given, observational F8 value"""
    # F8 is expected in units of ppt
    F8 = F8 / 1000
    
    σ = calc_σ_from_F8(logg, Teff, Z, F8, beta=beta)
    M = M * M_sun_grams
    prefix = 12 / np.sqrt(2) / np.sqrt(2*np.pi*G)
    factor = prefix * np.sqrt(tau_g(Teff, logg, Z, beta)) * Lambda(Teff, logg, beta) * np.sqrt(10**logg) / np.sqrt(M)
    theta = np.sqrt(σ / factor)
    return theta

def calc_F8_max(logg, Teff, M, Z, Ma=None, F8_fcn=calc_F8):
    """Calculates the maximum possible F8 according to the envelope"""
    if Ma is None:
        Ma = calc_Ma(logg, Teff, Z)
    phi = calc_theta_max(Ma) / calc_theta(Ma_sun)
    return F8_fcn(logg, Teff, M, Z, phi, Ma)

def calc_F8_min(logg, Teff, M, Z, Ma=None, F8_fcn=calc_F8):
    """Calculates the minimum possible F8 according to the envelope"""
    if Ma is None:
        Ma = calc_Ma(logg, Teff, Z)
    phi = calc_theta_min(Ma) / calc_theta(Ma_sun)
    return F8_fcn(logg, Teff, M, Z, phi, Ma)

def calc_F8_ratio_to_envelope(F8_emp, logg, Teff, M, Z, sig_mult=1, Ma_mult=1, Ma=None, F8_fcn=calc_F8):
    """Computes obs-to-model ratios using our envelope fit.
    
    Takes in observational F8s.  Returns 1 if that F8 is within the range of
    the F8 values given by our envelope fit, given a model-predicted Mach
    number. Otherwise returns the ratio of the empirical F8 to the nearest
    edge of the envelope range.
    """
    if Ma is None:
        Ma = calc_Ma(logg, Teff, Z) * Ma_mult
    bound_lower = calc_F8_min(logg, Teff, M, Z, Ma, F8_fcn=F8_fcn) * sig_mult
    bound_upper = calc_F8_max(logg, Teff, M, Z, Ma, F8_fcn=F8_fcn) * sig_mult
    ratio_lower = F8_emp / bound_lower
    ratio_upper = F8_emp / bound_upper
    out = np.ones_like(ratio_lower)
    out[F8_emp < bound_lower] = ratio_lower[F8_emp < bound_lower]
    out[F8_emp > bound_upper] = ratio_upper[F8_emp > bound_upper]
    return out



# ---- Plotting-oriented utility functions ----



def plot_outline(newer_version=True):
    """Plots the outline of Cranmer+2014 Figure 4"""
    x = np.linspace(.015, .34, 100)
    
    if newer_version:
        poly_fit = lambda x: 1.3724221 - 3.5002686*np.log10(x) - 1.6838185*np.log10(x)**2 - 0.37909094*np.log10(x)**3
    else:
        poly_fit = lambda x: 1.15136 - 3.59637*np.log10(x) - 1.40002*np.log10(x)**2 - 0.22993*np.log10(x)**3
        
    plt.plot(x, poly_fit(x) + 0.2, color='black')
    plt.plot(x, poly_fit(x) - 0.2, color='black')
    
    plt.plot([.015, .015], [poly_fit(.015)+.2, poly_fit(.015)-.2], color='black')
    plt.plot([.34, .34], [poly_fit(.34)+.2, poly_fit(.34)-.2], color='black')
    
    plt.gca().invert_yaxis()
    
def outline_data(x=None, y=None, cat=None, smooth=True, **kwargs):
    """Draws an outline of a set of points---by default, our stars in (T, logg)
    
    Accepts two one-dimentional arrays containing the x and y coordinates
    of the data set to be outlined.
    All kwargs are passed to plt.contour
    
    Smoothing will attempt to fill in holes, trim fuzz, and remove islands"""
    if cat is None:
        cat = catalog
    if x is None:
        x = cat['TeffH']
    if y is None:
        y = cat['loggH']
    
    H, x_edge, y_edge = np.histogram2d(x, y, bins=100)
    # H needs to be transposed for plt.contour
    H = H.T
    
    # Contour plotting wants the x & y arrays to match the
    # shape of the z array, so work out the middle of each bin
    x_edge = (x_edge[1:] + x_edge[:-1]) / 2
    y_edge = (y_edge[1:] + y_edge[:-1]) / 2
    
    H = H.astype(bool)
    if smooth:
        # This needs to happen before the padding is added
        H = ~scipy.ndimage.binary_fill_holes(~H)
    
    # Pad the arrays
    H = np.pad(H, 1)
    dx = x_edge[1] - x_edge[0]
    x_edge = np.insert(x_edge, 0, x_edge[0] - dx)
    x_edge = np.append(x_edge, x_edge[-1] + dx)
    dy = y_edge[1] - y_edge[0]
    y_edge = np.insert(y_edge, 0, y_edge[0] - dy)
    y_edge = np.append(y_edge, y_edge[-1] + dy)
    
    if smooth:
        # This needs to happen after the padding is added
        H = scipy.ndimage.binary_fill_holes(H)
        
        H = scipy.ndimage.binary_opening(H, iterations=1)
    
    XX, YY = np.meshgrid(x_edge, y_edge)
    
    # Fill in some default plot args if not given
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5
    if "colors" not in kwargs:
        kwargs["colors"] = "black"
    if "color" in kwargs:
        kwargs["colors"] = kwargs["color"]
    
    plt.contour(XX, YY, H, levels=[0.5], **kwargs)

def prep_2d_bins(cat, quantity, stat='median', binsize=100):
    """Utility function for plot_quasi_hi"""
    stat, r, c, binn = scipy.stats.binned_statistic_2d(
        cat['loggH'], cat['TeffH'], quantity, stat, binsize,
        expand_binnumbers=True,
        range=[[2.5, cat['loggH'].max()], [cat['TeffH'].min(), 7500]])
    return stat, r, c, binn

def plot_quasi_hr(cat, quantity, label=None, cmap="viridis", binsize=100,
                  vmin=None, vmax=None, scale_fcn=lambda x: x,
                  stat='median', log_norm=False, show_x_label=True,
                  show_y_label=True, imshowargs=None,
                  stat_in_log_space=False,
                  fill_in_bg=True, show_colorbar=True,
                  return_bins=False, return_cbar=False,
                  cbar_kwargs={}):
    """Plots a 2D histogram of a quantity in (T, logg) space, with axes flipped
    for HR-alike display.
    
    The first argument is a catalog (such as the one returned by load_catalog)
    containing loggH and TeffH columns, used to determine the x and y positions
    of each star.
    
    The second argument is the quantity being plotted in this 2D histogram, and
    should be a list/array of values, one for each star.
    
    The third argument is the label to put on the colorbar.
    
    The histogram is produced by default by taking the median of all values
    falling into each bin. This can be adjusted with the `stat` argument, which
    accepts any value recognized by scipy.stats.binned_statistic_2d.
    
    The histogram grid size is controlled by the `binsize` argument.
    
    The colorbar is controlled by the `cmap`, `vmin`, and `vmax` arguments.
    
    If log_norm is set to True, the color mapping will be done in logarithmic
    space.
    
    See analysis_and_plots.ipynb for examples of the use of this function."""
    if imshowargs is None:
        imshowargs = {}
    if log_norm:
        imshowargs['norm'] = LogNorm(vmin=vmin, vmax=vmax)
        vmin = None
        vmax = None
    
    if stat_in_log_space:
        quantity = np.log10(quantity)
    
    stat, r, c, binn = prep_2d_bins(cat, quantity, stat, binsize)    
    
    if stat_in_log_space:
        stat = 10**stat
    
    if fill_in_bg:
        plt.gca().set_facecolor('black')
    
    im = plt.imshow(scale_fcn(stat),
                    extent=(c.min(), c.max(), r.max(), r.min()),
                    vmin=vmin, vmax=vmax,
                    aspect='auto', cmap=cmap,
                    interpolation='nearest',
                    **imshowargs)
    
    if show_x_label:
        # textrm looks better, but it's only supported with
        # rcParams['text.usetex'] = True 
        if rcParams['text.usetex']:
            plt.xlabel(r"$T_\textrm{eff}$ (K)")
        else:
            plt.xlabel(r"$T_\mathrm{eff}$ (K)")
    if show_y_label:
        plt.ylabel("$\log\ g$")
    plt.xlim(plt.xlim()[1], plt.xlim()[0])
    
    if show_colorbar:
        cb = plt.colorbar(**cbar_kwargs)
        cb.set_label(label)
        if return_cbar:
            return cb
    if return_bins:
        return stat, r, c, binn
    return im
