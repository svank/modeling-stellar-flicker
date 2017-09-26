#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from base import *

evo_data = np.loadtxt("evolutionary_status/table1.dat", dtype=np.string_, unpack=True)
KIC = evo_data[0].astype(np.int32)
Dnu, e_Dnu, DPi1, e_DPi1 = evo_data[1:5].astype(np.float64)
Stat = evo_data[5]
Mass, e_Mass = evo_data[6:].astype(np.float64)

phi_vals = fit_Φ(catalog['loggH'],
                 catalog['TeffH'],
                 catalog['MH'],
                 catalog['F8'])

phi_lists = dict()
logg_lists = dict()
f8_lists = dict()
for stat in Stat:
    phi_lists[stat] = []
    logg_lists[stat] = []
    f8_lists[stat] = []

for i in range(len(KIC)):
    if KIC[i] not in catalog['KIC']:
        print("{} missing".format(KIC[i]))
        continue

    phi = phi_vals[catalog['KIC'] == KIC[i]]
    logg = catalog['loggH'][catalog['KIC'] == KIC[i]]
    f8 = catalog['F8'][catalog['KIC'] == KIC[i]]
    phi_lists[Stat[i]].append(phi)
    logg_lists[Stat[i]].append(logg)
    f8_lists[Stat[i]].append(f8)

counter = 0
categories = ['S', 'R', 'f', 'C', 'p2', '0', 'A', '?']
for key in categories:
    counter += 1
    key = np.array([key], dtype=np.string_)[0]
    if key not in phi_lists.keys():
        print("{:5}, n=0".format(key))
        continue

    n = len(phi_lists[key])
    mean_phi = np.mean(phi_lists[key])
    std_phi = np.std(phi_lists[key])
    mean_logg = np.mean(logg_lists[key])
    std_logg = np.std(logg_lists[key])
    print("{:5}, n={:3d}, μ_phi={:5f}, σ_phi={:5f}, μ_logg={:5f}, σ_logg={:5f}".format(key, n, mean_phi, std_phi, mean_logg, std_logg))
    plt.subplot(121)
    plt.errorbar(counter, mean_phi, yerr=std_phi, fmt='o')

    plt.subplot(122)
    plt.errorbar(counter, mean_logg, yerr=std_logg, fmt='o')

plt.subplot(121)
plt.xlim(0, counter+1)
plt.xlabel("Evolutionary Stage")
plt.ylabel("Fitted $\Phi$")
plt.title("Evolutionary Stages of Mosser et al. (2014)")
plt.xticks(range(1, 1+len(categories)), categories)

plt.subplot(122)
plt.xlim(0, counter+1)
plt.xlabel("Evolutionary Stage")
plt.ylabel("log(g)")
plt.title("Evolutionary Stages of Mosser et al. (2014)")
plt.xticks(range(1, 1+len(categories)), categories)


plt.gcf().set_size_inches(12, 5)
plt.tight_layout()

plt.savefig("evo_plot.png", dpi=200)
