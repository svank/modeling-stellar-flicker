#!/usr/bin/env python3

from base import *

c = catalog[ catalog['has_C'] == 1 ]

S = fit_S(c['loggC'], c['TeffC'], c['MC'], c['F8'])

x = np.linspace(4500, 7000, 51)
y = calc_cranmer_S(x)
plt.plot(x, y)
plt.scatter(c['TeffC'], S)
plt.gca().invert_xaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetic Suppression Factor $S$")
plt.title("Fitted $S$ for Cranmer et al. (2014) stars")
plt.savefig('cf_cranmer.png', dpi=200)
plt.show()


c = catalog[ catalog['TeffH'] < 7000]

S = fit_S(c['loggH'], c['TeffH'], c['MH'], c['F8'])

print("{} good values".format(np.sum(S > 0)))
print("{} bad values".format(np.sum(S <= 0) + np.sum(np.isnan(S))))
print("{} total values".format(len(S)))

x = np.linspace(4500, 7000, 100)
y = calc_cranmer_S(x)
plt.plot(x, y, linewidth=5, color='white')
plt.hexbin(c['TeffH'], S, cmap='hot')
plt.colorbar()
plt.gca().invert_xaxis()
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetic Suppression Factor $S$")
plt.title("Fitted $S$ for Bastien et al. (2016) stars")
plt.savefig('new_result.png', dpi=200)
plt.show()

