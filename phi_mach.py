#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from base import *
import colormaps

cat = catalog[ catalog['TeffH'] < 7000 ]
Φ = fit_Φ(cat['loggH'], cat['TeffH'], cat['MH'], cat['F8'])
Ma = calc_Ma(cat['loggH'], cat['TeffH'])

plt.hexbin(Ma, Φ, bins='log')
plt.colorbar().set_label("log(Count)")

model_Ma = np.linspace(.125, .7125, 200)
model_Φ = calc_Φ(model_Ma)
plt.plot(model_Ma, model_Φ, linewidth=5, color='white')

plt.xlabel("Mach Number")
plt.ylabel("Fitted $\Phi$")
plt.title("Samadi's Model Versus Real Life Fitted $\Phi$")

#plt.show()
plt.savefig("phi-versus-Ma.png", dpi=200)
