#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
from progressBar import ProgressBar
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

M_sun = 1 # Solar masses
T_sun = 5777 # K
logg_sun = 4.438 # log-cgs units, I suppose
ν_sun = 3.106e-3 # Hz


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
    Ma = calc_Ma(logg, Teff)
    Ma *= S
    if Φ is None:
        # TODO: Why is the phi(.26) here? It's not in the paper
        Φ = calc_Φ(Ma) / calc_Φ(.26)
        
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


def calc_Φ(Ma, make_monotonic=True):
    """Calculates the temperature fluctuation amplitude
    Cranmer 2014's Eqn 4
    """
    
    # TODO: Check in on these---numbers might be reported wrong in Steve's paper,
    # but calculations behind his plots should be fine
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
    Ma_sun = 0.26 # As given in Cranmer 2014
    
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
            Φ[i] = scipy.optimize.newton(obj, x0=3)
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


#merge_catalog()

catalog = load_catalog()

catalog['F8'] = F8_from_logg(catalog['F8logg'])

#F8 = F8_from_logg(F8logg)

#σ = calc_σ(Teff, M, loggH)
#F8M = calc_F8_from_σ(loggH, Teff, σ)

#plt.subplot(211)
#plt.hexbin(F8, loggH, cmap='hot')
#plt.title("Observed")
#plt.colorbar()
#plt.gca().invert_yaxis()
#plt.xlim(0, .5)

#plt.subplot(212)
#plt.hexbin(F8M, loggH, cmap='hot')
#plt.title("Modeled")
#plt.colorbar()
#plt.gca().invert_yaxis()
#plt.xlim(0, .5)

#plt.tight_layout()
#plt.show()
