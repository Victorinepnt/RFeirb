import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def divide_normal_distributions(mu1, sigma1, mu2, sigma2, x):
    # Calcul des paramètres de la loi normale résultante
    mu_result = mu1 / mu2
    sigma_result = np.sqrt(sigma1**2 / sigma2**2)

    # Calcul de la fonction de densité de probabilité de la loi normale résultante
    result = norm.pdf(x, mu_result, sigma_result)

    return result

# Paramètres de la première loi normale
mu1 = 0
sigma1 = 1

# Paramètres de la deuxième loi normale
mu2 = 2
sigma2 = 1.5

# Définir une plage de valeurs x
x = np.linspace(-5, 5, 1000)

# Calculer la division des lois normales
result_distribution = divide_normal_distributions(mu1, sigma1, mu2, sigma2, x)

# Afficher les résultats
plt.plot(x, norm.pdf(x, mu1, sigma1), label='N(0, 1)')
plt.plot(x, norm.pdf(x, mu2, sigma2), label='N(2, 1.5)')
plt.plot(x, result_distribution, label='N(0/2, 1^2/1.5^2)')

plt.title('Division de deux lois normales')
plt.legend()
plt.show()
