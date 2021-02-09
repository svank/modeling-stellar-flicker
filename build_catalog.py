#!/usr/bin/env python3

"""Matches the data from our disparate sources and saves a merged catalog.

Can be simply invoked from the command line: `python build_catalog.py`
"""
    

import numpy as np

# This will needlessly load the catalog, Aesopus data, etc. but ¯\_(ツ)_/¯
from base import F8_from_logg

def build_catalog():
    # We used to use Huber+2014, thus the name
    huber_raw = np.genfromtxt("orig_data/berger_2020.txt", skip_header=45,
            usecols=range(13))
    
    huber = dict()
    for i in range(huber_raw.shape[0]):
        row = huber_raw[i]
        huber[int(row[0])] = row
    
    cranmer_raw = np.genfromtxt("orig_data/Cranmer_2014.txt", skip_header=1, skip_footer=16)
    
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
    
    (KIC, kepmag, F8logg, E_F8logg, e_F8logg, Range, RMS, Teff) = bastien_raw
    
    lamost_raw = np.genfromtxt("orig_data/LAMOST_Kepler_ApJS2018_spectra_list.out", skip_header=39,
            delimiter=(10, 12, 17, 15, 15, 10, 16, 8, 21, 19, 11, 3, 6, 20, 23, 12, 5)) 
    lamost = dict()
    for i in range(lamost_raw.shape[0]):
        row = lamost_raw[i]
        if not np.isnan(row[1]) and not np.isnan(row[10]) and not np.isnan(row[12]):
            lamost[int(row[1])] = row
    
    # Catalog entries are grouped by source.
    # Note that, for some quantites appearing in multiple source catalogs,
    # suffixes in the column name distinguish values from those multiple sources.
    # The 'has_*' columns store whether a given star has data from a given
    # source. Columns can be accessed by name. E.g.:
    #   selection = catalog['TeffH'] > 5770
    #   catalog['loggH'][selection]
    #   catalog[selection]['loggH']
    # After executing the first line, executing either the second or the third
    # line (the two are interchangeable) will provide logg values for all stars
    # with temperatures greater than 5770. Both quantities are drawn from the
    # 'H' source (see below).
    catalog = np.zeros(len(KIC), dtype=[
                   ('KIC', 'int32'),
                   
                   # These come from Bastien+2016
                   ('kepmag', 'float64'),
                   ('F8logg', 'float64'),
                   ('E_F8logg', 'float64'),
                   ('e_F8logg', 'float64'),
                   ('F8', 'float64'),
                   ('Range', 'float64'),
                   ('RMS', 'float64'),
                   ('Teff', 'float64'),
                   
                   # These come from Berger+2020. We used to use Huber+2014,
                   # thus the 'H' suffix
                   ('TeffH', 'float64'),
                   ('E_TeffH', 'float64'),
                   ('e_TeffH', 'float64'),
                   ('loggH', 'float64'),
                   ('E_loggH', 'float64'),
                   ('e_loggH', 'float64'),
                   ('MH', 'float64'),
                   ('E_MH', 'float64'),
                   ('e_MH', 'float64'),
                   ('FeHH', 'float64'),
                   ('E_FeHH', 'float64'),
                   ('e_FeHH', 'float64'),
                   ('has_H', 'bool_'),
                   
                   # For those few stars in their sample, the Cranmer+2014 data
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
                   
                   # The Lamost data. (The citation is Zong+2018)
                   ('FeH', 'float64'),
                   ('e_FeH', 'float64'),
                   ('has_L', 'bool_'),
                   
                   # McQuillan+2014 rotation data
                   ('PRot', 'float64'),
                   ('e_PRot', 'float64'),
                   ('RPer', 'float64'),
                   ('PFlag', 'str_'),
                   ('has_M', 'bool_'),
                   
                   # Zhang+2020's magnetic activity data
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
    
    for i, kic in enumerate(KIC):
        if kic in huber:
            catalog['has_H'][i] = 1
            h_data = huber[kic]
            catalog['TeffH'][i]   = h_data[4]
            catalog['E_TeffH'][i] = h_data[5]
            catalog['e_TeffH'][i] = h_data[6]
            catalog['loggH'][i]   = h_data[7]
            catalog['E_loggH'][i] = h_data[8]
            catalog['e_loggH'][i] = h_data[9]
            catalog['MH'][i]      = h_data[1]
            catalog['E_MH'][i]    = h_data[2]
            catalog['e_MH'][i]    = h_data[3]
            catalog['FeHH'][i]    = h_data[10]
            catalog['E_FeHH'][i]  = h_data[11]
            catalog['e_FeHH'][i]  = h_data[12]
        
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
            
    catalog['F8'] = F8_from_logg(catalog['F8logg'])
    
    np.save('merged_catalog.npy', catalog)

if __name__ == "__main__":
    build_catalog()
