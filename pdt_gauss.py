import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma_squared):
   return (1.0 / np.sqrt(2.0 * np.pi * sigma_squared)) * np.exp(-0.5 * ((x - mu)**2 / sigma_squared))

def multiply_gaussians(mu1, sigma1_squared, mu2, sigma2_squared):

    sigma_squared = 1.0 / (1.0 / sigma1_squared + 1.0 / sigma2_squared)
    mu = sigma_squared * (mu1 / sigma1_squared + mu2 / sigma2_squared)
    return mu, sigma_squared

# Paramètres des gaussiennes
mu1 = 1.0
sigma1_squared = 2.0
mu2 = 3.0
sigma2_squared = 4.0

# Calcul de la gaussienne résultante
mu_result, sigma_squared_result = multiply_gaussians(mu1, sigma1_squared, mu2, sigma2_squared)

# Points x pour les tracés
x = np.linspace(-10, 10, 1000)

# Tracés des gaussiennes
plt.plot(x, gaussian(x, mu1, sigma1_squared), label='Gaussienne 1')
plt.plot(x, gaussian(x, mu2, sigma2_squared), label='Gaussienne 2')
plt.plot(x, gaussian(x, mu_result, sigma_squared_result), label='Produit des Gaussiennes', linestyle='--')

plt.title('Multiplication de deux Gaussiennes')
plt.xlabel('x')
plt.ylabel('Densité de probabilité')
plt.legend()
plt.show()

