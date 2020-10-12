#!/usr/bin/env python3

import numpy as np
from progressBar import ProgressBar

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

def build_catalog():
    huber_raw = np.loadtxt("orig_data/Huber_2014.txt", skiprows=49, usecols=range(19))
    
    huber = dict()
    for i in range(huber_raw.shape[0]):
        row = huber_raw[i]
        huber[int(row[0])] = row
    
    cranmer_raw = np.loadtxt("orig_data/Cranmer_2014.txt", skiprows=30)
    
    cranmer = dict()
    for i in range(cranmer_raw.shape[0]):
        row = cranmer_raw[i]
        cranmer[int(row[0])] = row
    
    mcquillan_raw = np.genfromtxt("orig_data/McQuillan_2014.txt", skip_header=32)
    
    mcquillan = dict()
    for i in range(mcquillan_raw.shape[0]):
        row = mcquillan_raw[i]
        mcquillan[int(row[0])] = row
        
    
    zhang_raw = np.genfromtxt("orig_data/Zhang_2020.txt", skip_header=25)
    
    zhang = dict()
    for i in range(zhang_raw.shape[0]):
        row = zhang_raw[i]
        zhang[int(row[0])] = row
        
    bastien_raw = np.loadtxt("orig_data/Bastien_2016.txt", skiprows=27, unpack=True)
    
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
                   ('S', 'float64'),
                   ('e_S', 'float64'),
                   ('logR+HK', 'float64'),
                   ('e_logR+HK', 'float64'),
                   ('Reff', 'float64'),
                   ('e_Reff', 'float64'),
                   ('has_Z', 'bool_')])
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
        
        if kic in zhang:
            catalog['has_Z'][i] = 1
            z_data = zhang[kic]
            catalog['S'][i] = float(z_data[7])
            catalog['e_S'][i] = float(z_data[8])
            catalog['logR+HK'][i] = float(z_data[9])
            catalog['e_logR+HK'][i] = float(z_data[10])
            catalog['Reff'][i] = float(z_data[11])
            catalog['e_Reff'][i] = float(z_data[12])
            
        pb.increment()
        pb.display()
    pb.display(True)
    
    catalog['F8'] = F8_from_logg(catalog['F8logg'])
    
    np.save('merged_catalog.npy', catalog)

if __name__ == "__main__":
    build_catalog()
