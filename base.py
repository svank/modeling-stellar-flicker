#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import astropy.constants as consts
from progressBar import ProgressBar
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

M_sun = 1 # Solar masses
M_sun_grams = 1.998e33
T_sun = 5777 # K
logg_sun = 4.438 # log-cgs units, I suppose
ν_sun = 3.106e-3 # Hz
Ma_sun = 0.26 # As given in Cranmer 2014
Z_sun = 0.016
R_sun_cm = 6.957e10


def merge_catalog():
    huber_raw = np.loadtxt("Huber_2014.txt", skiprows=49, usecols=range(19))
    
    huber = dict()
    for i in range(huber_raw.shape[0]):
        row = huber_raw[i]
        huber[int(row[0])] = row
    
    cranmer_raw = np.loadtxt("Cranmer_2014.txt", skiprows=30)
    
    cranmer = dict()
    for i in range(cranmer_raw.shape[0]):
        row = cranmer_raw[i]
        cranmer[int(row[0])] = row
    
    mcquillan_raw = np.genfromtxt("orig_data/McQuillan_2014.txt", skip_header=32)
    
    mcquillan = dict()
    for i in range(mcquillan_raw.shape[0]):
        row = mcquillan_raw[i]
        mcquillan[int(row[0])] = row
        
        
    bastien_raw = np.loadtxt("Bastien_2016.txt", skiprows=27, unpack=True)
    
    lamost_raw = np.genfromtxt("orig_data/lamost_dr5_vac.out", skip_header=39,
            delimiter=(10, 12, 17, 15, 15, 10, 16, 8, 21, 19, 11, 3, 6, 20, 23, 12, 5)) 
    lamost = dict()
    for i in range(lamost_raw.shape[0]):
        row = lamost_raw[i]
        if not np.isnan(row[1]) and not np.isnan(row[10]) and not np.isnan(row[12]):
            lamost[int(row[1])] = row
    
    (KIC, kepmag, F8logg, E_F8logg, e_F8logg, Range, RMS, Teff) = bastien_raw
    
    catalog=np.zeros(len(KIC), dtype=[
                   ('KIC', 'int32'),
                   ('kepmag', 'float64'),
                   ('F8logg', 'float64'),
                   ('E_F8logg', 'float64'),
                   ('e_F8logg', 'float64'),
                   ('F8', 'float64'),
                   ('Range', 'float64'),
                   ('RMS', 'float64'),
                   ('Teff', 'float64'),
                   ('TeffH', 'float64'),
                   ('E_TeffH', 'float64'),
                   ('e_TeffH', 'float64'),
                   ('loggH', 'float64'),
                   ('E_loggH', 'float64'),
                   ('e_loggH', 'float64'),
                   ('MH', 'float64'),
                   ('E_MH', 'float64'),
                   ('e_MH', 'float64'),
                   ('TeffC', 'float64'),
                   ('loggC', 'float64'),
                   ('MC', 'float64'),
                   ('tau_effC', 'float64'),
                   ('MaC', 'float64'),
                   ('sigmaC', 'float64'),
                   ('F8_modC', 'float64'),
                   ('F8_obsC', 'float64'),
                   ('RvarC', 'float64'),
                   ('has_C', 'bool_'),
                   ('FeH', 'float64'),
                   ('e_FeH', 'float64'),
                   ('has_L', 'bool_'),
                   ('PRot', 'float64'),
                   ('e_PRot', 'float64'),
                   ('RPer', 'float64'),
                   ('PFlag', 'str_'),
                   ('has_M', 'bool_'),
                   ('myF8', 'float64')])
    catalog['KIC'] = KIC
    catalog['kepmag'] = kepmag
    catalog['F8logg'] = F8logg
    catalog['E_F8logg'] = E_F8logg
    catalog['e_F8logg'] = e_F8logg
    catalog['Range'] = Range
    catalog['RMS'] = RMS
    catalog['Teff'] = Teff
    
    pb = ProgressBar(len(KIC))
    for i, kic in enumerate(KIC):
        h_data = huber[kic]
        catalog['TeffH'][i]   = h_data[1]
        catalog['E_TeffH'][i] = h_data[2]
        catalog['e_TeffH'][i] = h_data[3]
        catalog['loggH'][i]   = h_data[4]
        catalog['E_loggH'][i] = h_data[5]
        catalog['e_loggH'][i] = h_data[6]
        catalog['MH'][i]      = h_data[13]
        catalog['E_MH'][i]    = h_data[14]
        catalog['e_MH'][i]    = h_data[15]
        
        if kic in cranmer:
            catalog['has_C'][i] = 1
            c_data = cranmer[kic]
            catalog['TeffC'][i] = c_data[1]
            catalog['loggC'][i] = c_data[2]
            catalog['MC'][i] = c_data[3]
            catalog['tau_effC'][i] = c_data[5]
            catalog['MaC'][i] = c_data[6]
            catalog['sigmaC'][i] = c_data[7]
            catalog['F8_modC'][i] = c_data[8]
            catalog['F8_obsC'][i] = c_data[9]
            catalog['RvarC'][i] = c_data[10]
        
        if kic in lamost:
            catalog['has_L'][i] = 1
            l_data = lamost[kic]
            catalog['FeH'][i] = float(l_data[10])
            catalog['e_FeH'][i] = float(l_data[12])
        
        if kic in mcquillan:
            catalog['has_M'][i] = 1
            m_data = mcquillan[kic]
            catalog['PRot'][i] = float(m_data[4])
            catalog['e_PRot'][i] = float(m_data[5])
            catalog['RPer'][i] = float(m_data[6])
            catalog['PFlag'][i] = m_data[10]
        pb.increment()
        pb.display()
    pb.display(True)
    np.save('merged_catalog.npy', catalog)


def load_catalog(fname='merged_catalog.npy'):
    return np.load(fname)


def F8_from_logg(logg_array):
    """Calculates F8 flicker values from log(g)
    Bastien 2016's Eqn 4.
    
    log g = 1.3724221 - 3.5002686 x - 1.6838185 x^2 - 0.37909094 x^3
    x = log10(F8)
    """
    F8 = np.zeros_like(logg_array)
    for i, logg in enumerate(logg_array):
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
        
        F8[i] = 10**root
    return F8


def calc_σ(Teff, M, logg, S=1, Φ=None):
    """Calculates RMS amplitude σ of photospheric continuum intensity variations
    Cranmer 2014's Eqn 1
    """
    ν_max = calc_ν_max(logg, Teff)
    if Φ is None:
        Ma = calc_Ma(logg, Teff)
        Ma *= S
        
        Φ = calc_Φ(Ma)
        
        # Clamp Φ at zero
        Φ = 0 * (Φ < 0) + Φ * (Φ >= 0)
    
    # Why 1.03 power when Samadi has 1.10? See Samadi Figure B.1
    return 0.039 * ( (Teff / T_sun) ** (3/4)
                     * (M_sun * ν_sun / M / ν_max) ** (1/2)
                     * Φ**2 )**1.03


def calc_ν_max(logg, Teff):
    """Calculates peak p-mode frequency
    Cranmer 2014's Eqn 2
    """
    return ν_sun * 10**logg / 10**logg_sun * (T_sun / Teff)**0.5


def calc_Ma(logg, Teff):
    """Calculates Mach number
    Cranmer 2014's Eqn 3
    """
    return 0.26 * (Teff / T_sun)**2.35 * (10**logg_sun / 10**logg) ** 0.152


def calc_phi(*args, **kwargs):
    return calc_Φ(*args, **kwargs)


def calc_Φ(Ma, make_monotonic=True):
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


def calc_F8_from_σ(logg, Teff, σ, S=1):
    """Cranmer 2014's Eqn 8"""
    ν_8 = 1 / (8 * 3600)
    ν_max = calc_ν_max(logg, Teff)
    Ma = calc_Ma(logg, Teff) * S
    
    τ_eff = 300 * (ν_sun * Ma_sun / ν_max / Ma)**0.98
    
    return σ * np.sqrt(1 - 2 / np.pi * np.arctan(4 * τ_eff * ν_8))


def calc_F8(logg, Teff, M, S=1, Φ=None):
    σ = calc_σ(Teff, M, logg, S, Φ)
    return calc_F8_from_σ(logg, Teff, σ, S)


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
        obj = lambda S: calc_F8(logg, Teff, M, S) - F8obs
        
        try:
            S[i] = scipy.optimize.newton(obj, x0=calc_cranmer_S(Teff))
        except:
            S[i] = float('nan')
            #s0 = calc_cranmer_S(Teff)
            #print(i, s0, calc_F8(logg, Teff, M, s0), calc_F8(logg, Teff, M, 1), F8obs)
    
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
        obj = lambda Φ: calc_F8(logg, Teff, M, Φ=Φ) - F8obs
        
        try:
            Φ[i] = scipy.optimize.newton(obj, x0=1)
        except:
            Φ[i] = float('nan')
    
    return Φ


def plot_outline(newer_version=True):
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
    
def plot_ZAMS_Teff_logg(**kwargs):
    M, R, L, T, T_core, rho_core = np.genfromtxt(
            "orig_data/zams_star_properties.dat.txt",
            skip_header=10, unpack=True)
    from astropy.constants import G
    M *= M_sun_grams
    R *= R_sun_cm
    
    g = G.cgs.value * M / R**2
    
    plt.plot(T, np.log10(g), **kwargs)

def calc_N_gran(R, Teff, logg):
    Lambda = (Teff / T_sun) / (10**logg / 10**logg_sun) * 1e8 # gran size, cm
    return 2 * np.pi * R**2 / Lambda**2


# ---- New Versions ----



def calc_Φ_from_F8_new(F8, logg, Teff, M, Z):
    σ = calc_σ_from_F8_new(logg, Teff, Z, F8)
    ν_max = calc_ν_max(logg, Teff)
    factor = (Teff / T_sun) ** (3/4) * (M_sun * ν_sun / M / ν_max) ** (1/2)
    return ( (σ / 0.039)**(1/1.10) / factor ) ** 0.5
calc_phi_from_F8_new = calc_Φ_from_F8_new

def calc_phi_new(Ma):
    A = 1.62191904e+00
    B = 1.59965328e+01
    e2 = -1.03057868e-02
    return 1 / (A * Ma ** (-2) + B * Ma ** (e2)) ** (1)

def old_calc_phi_new(Ma):
    a = 0.15390654
    b = 0.08877965
    M0 = 0.32299999
    
    phi0 = a*M0**2 + b*M0
    return (a*Ma**2 + b*Ma) * (Ma < M0) + phi0 * (Ma >= M0)

def calc_F8_new(logg, Teff, M, Z):
    σ = calc_σ_new(Teff, M, logg, Z)
    return calc_F8_from_σ_new(logg, Teff, Z, σ)

def calc_σ_new(Teff, M, logg, Z):
    """Calculates RMS amplitude σ of photospheric continuum intensity variations
    Cranmer 2014's Eqn 1
    """
    ν_max = calc_ν_max(logg, Teff)
    Ma = calc_Ma_new(logg, Teff, Z)
    Φ = calc_phi_new(Ma)
    Φ_sun = calc_phi_new(Ma_sun)
    
    # Steve had a power of 1.03, drawing from Samadi's appendix. But that
    # exponent doesn't actually apply to this equation, so return to Samadi's
    # 1.10
    return 0.039 * ( (Teff / T_sun) ** (3/4)
                     * (M_sun * ν_sun / M / ν_max) ** (1/2)
                     * (Φ / Φ_sun) ** 2)**1.10

def calc_F8_from_σ_new(logg, Teff, Z, σ):
    """Cranmer 2014's Eqn 8"""
    ν_8 = 1 / (8 * 3600)
    ν_max = calc_ν_max(logg, Teff)
    Ma = calc_Ma_new(logg, Teff, Z)
    
    τ_eff = 300 * (ν_sun * Ma_sun / ν_max / Ma)**0.98
    
    return σ * np.sqrt(1 - 2 / np.pi * np.arctan(4 * τ_eff * ν_8))

def calc_Ma_new(logg, Teff, Z):
    rho_sun = find_rho(T_sun, Z_sun, logg_sun) 
    
    rho = find_rho(Teff, Z, logg)
    return Ma_sun * (Teff / T_sun) ** (5/6) * (rho / rho_sun) ** (-1/3)

aesopus_data = None

def load_aesopus():
    global aesopus_R_vals, aesopus_data
    global aesopus_T_idx_mapper, aesopus_Z_idx_mapper
    from pathlib import Path
    
    files = list(Path("orig_data/aesopus/").glob("Z*"))
    aesopus_data = np.zeros((len(files), 67, 91))
    aesopus_Z_vals = np.zeros(len(files))
    aesopus_R_vals = 10**np.arange(-8, 1.01, .1)
    for i, f in enumerate(sorted(files)):
        aesopus_Z_vals[i] = str(f).split("Z")[1]
        # print(f, aesopus_Z_vals[i])
        d = np.genfromtxt(f, skip_header=329)
        aesopus_T_vals = 10**d[:, 0]
        aesopus_data[i] = d[:, 1:] 
    
    import scipy.interpolate
    aesopus_T_idx_mapper = scipy.interpolate.interp1d(aesopus_T_vals, np.arange(aesopus_T_vals.size), kind='linear')
    aesopus_Z_idx_mapper = scipy.interpolate.interp1d(aesopus_Z_vals, np.arange(aesopus_Z_vals.size), kind='linear') 

load_aesopus()

def _find_rho(T_val, Z_val, logg_val):
    k_B = consts.k_B.cgs.value
    m_h = consts.m_p.cgs.value
    
    T_idx = aesopus_T_idx_mapper(T_val)
    t1, t2 = int(np.floor(T_idx)), int(np.ceil(T_idx))
    t1r = 1 - (T_idx - t1)
    t2r = 1 - t1r

    Z_idx = aesopus_Z_idx_mapper(Z_val)
    z1, z2 = int(np.floor(Z_idx)), int(np.ceil(Z_idx))
    z1r = 1 - (Z_idx - z1)
    z2r = 1 - z1r

    κ = (  10**aesopus_data[z1, t1] * z1r * t1r
         + 10**aesopus_data[z1, t2] * z1r * t2r
         + 10**aesopus_data[z2, t1] * z2r * t1r
         + 10**aesopus_data[z2, t2] * z2r * t2r )

    ρ = aesopus_R_vals * (1e-6 * T_val)**3

    μ = 7/4 + .5 * np.tanh( (3500-T_val) / 600)
    H = k_B * T_val / μ / m_h / 10**logg_val

    τ = κ * ρ * H

    ρ_idx = scipy.interpolate.interp1d(τ, np.arange(τ.size), kind='linear', fill_value="extrapolate")(2/3)
    ρ_val = scipy.interpolate.interp1d(τ, ρ, kind='linear', fill_value="extrapolate")(2/3)

    κ_idx = scipy.interpolate.interp1d(τ, np.arange(κ.size), kind='linear', fill_value="extrapolate")(2/3)
    κ_val = scipy.interpolate.interp1d(τ, κ, kind='linear', fill_value="extrapolate")(2/3)
    
    # This version doesn't extrapolate
#     ρ_idx = np.interp(2/3, τ, np.arange(τ.size))
#     ρ_val = np.interp(2/3, τ, ρ)

#     κ_idx = np.interp(2/3, τ, np.arange(κ.size))
#     κ_val = np.interp(2/3, τ, κ)
    
    return ρ_val

find_rho = np.vectorize(_find_rho)

def FeH_to_Z(FeH):
    # TODO: Maybe still try to find what constant is baked into LAMOST's Fe/H
    # values. Steve had found that the solar Fe/H hasn't changed much over time,
    # so there's hope that what they used is what everyone else uses and we
    # don't need to find theirs specifically.
    return 0.01696 * 10**FeH

def calc_convective_turnover_time(Teff):
    return 0.002 + 314.24 * np.exp(-(Teff/1952.5) - (Teff/6250)**18)



# -----------------





#merge_catalog()

catalog = load_catalog()

catalog['F8'] = F8_from_logg(catalog['F8logg'])
