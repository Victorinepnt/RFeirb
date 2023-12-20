import numpy as np
import matplotlib.pyplot as plt

D = 100
sigma2 = 0.25

Z1 = 1.5 + np.random.randn(1, D) * np.sqrt(sigma2)
Z2 = -1.5 + np.random.randn(1, D) * np.sqrt(sigma2)

Z = np.zeros(D)
B = np.random.randint(1, 3, size=D)
Z[B == 1] = Z1[0, B == 1]
Z[B == 2] = Z2[0, B == 2]
    

z = np.linspace(-5, 5, 10000)

# Vraie loi
pz1 = np.exp(-(z - 1.5)**2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
pz2 = np.exp(-(z + 1.5)**2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
pz = 0.5 * (pz1 + pz2)
plt.plot(z, pz, linewidth=3)

# Loi approximée par histogramme
plt.hist(Z, bins=20, density=True, alpha=0.5)

# Loi approximée par méthode des noyaux
epsilon2 = 0.07
hpz = np.zeros_like(z)
for i in range(D):
    hpz = hpz + np.exp(-(z - Z[i])**2 / (2 * epsilon2)) / (D * np.sqrt(2 * np.pi * epsilon2))
plt.plot(z, hpz, linewidth=3)

# Loi approximée par méthode des noyaux avec la moyenne et l'écart-type
hmuz = np.mean(Z)
hstdz = np.std(Z)
hpz2 = np.exp(-(z - hmuz)**2 / (2 * hstdz**2)) / (np.sqrt(2 * np.pi) * hstdz)
plt.plot(z, hpz2, linewidth=3)

plt.show()
