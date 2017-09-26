#!/usr/bin/env python3

from base import *

c = catalog[ catalog['has_C'] == 1 ]

S = fit_S(c['loggC'], c['TeffC'], c['MC'], c['F8'])

#plt.subplot(221)
#plt.scatter(c['F8_obsC'], c['loggC'])
#plt.xlabel("F8")
#plt.ylabel("$\log\;g$")
#plt.xlim(0, .35)
#plot_outline()
#plt.title("Bastien et al. (2013) observed F8")

plt.subplot(221)
s = 1 + np.zeros_like(c['loggC'])
f = calc_F8(c['loggC'], c['TeffC'], c['MC'], s)
plt.scatter(f, c['loggC'])
plt.xlabel("$F_8$")
plt.ylabel("$\log\;g$")
plt.xlim(0, .35)
plot_outline()
plt.title("Model F8 with S=1")

plt.subplot(222)
plt.scatter(c['F8'], c['loggC'])
plt.xlabel("$F_8$")
plt.ylabel("$\log\;g$")
plt.xlim(0, .35)
plot_outline()
plt.title("Bastien et al. (2016) F8")

plt.subplot(223)
s = calc_cranmer_S(c['TeffC'])
f = calc_F8(c['loggC'], c['TeffC'], c['MC'], s)
plt.scatter(f, c['loggC'])
plt.xlabel("$F_8$")
plt.ylabel("$\log\;g$")
plt.xlim(0, .35)
plot_outline()
plt.title("Cranmer et al. (2014) F8")

plt.subplot(224)
plt.scatter(calc_F8(c['loggC'], c['TeffC'], c['MC'], S), c['loggC'])
plt.xlabel("$F_8$")
plt.ylabel("$\log\;g$")
plt.xlim(0, .35)
plot_outline()
plt.title("Fitted F8")

plt.tight_layout()
plt.savefig("compare_F8.png", dpi=200)
plt.show()
