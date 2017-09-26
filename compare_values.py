#!/usr/bin/env python3

from base import *

c = catalog[ catalog['has_C'] == 1]
cc = c[ c['TeffC'] < 5400 ]

Ma = calc_Ma(cc['loggC'], cc['TeffC'])
ν_max = calc_ν_max(cc['loggC'], cc['TeffC'])
Ma_sun = calc_Ma(logg_sun, T_sun)
τ_eff = 300 * (ν_sun * Ma_sun / ν_max / Ma)**0.98
σ = calc_σ(cc['TeffC'], cc['MC'], cc['loggC'])
F8 = calc_F8_from_σ(cc['loggC'], cc['TeffC'], σ)


plt.scatter(Ma, Ma - cc['MaC'])
plt.title("Ma")
plt.show()

plt.scatter(τ_eff, τ_eff - cc['tau_effC'])
plt.title("Tau")
plt.show()

plt.scatter(σ, σ - cc['σC'])
plt.title("sigma")
plt.show()

plt.scatter(F8, F8 - cc['F8_modC'])
plt.title("F8")
plt.show()

