# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:01:19 2023

@author: musav
"""

import numpy as np
import matplotlib.pyplot as plt

# Cas pour plusieurs signaux
# Paramètres
Fs = 1000          
T = 1/Fs           
fc = 100           
Ts = 1/(2*fc)      

# Génération des bits pour chaque signal
nb_signals = 5      # Nombre de signaux
nb_bits_per_signal = 1000
bits = np.random.randint(2, size=(nb_signals, nb_bits_per_signal))

# Mapping QPSK pour chaque signal
qpsk_symbols = np.zeros((nb_signals, nb_bits_per_signal//2), dtype=complex)
for j in range(nb_signals):
    for i in range(0, nb_bits_per_signal, 2):
        if bits[j, i] == 0 and bits[j, i+1] == 0:
            qpsk_symbols[j, i//2] = 1 + 1j   
        elif bits[j, i] == 0 and bits[j, i+1] == 1:
            qpsk_symbols[j, i//2] = -1 + 1j  
        elif bits[j, i] == 1 and bits[j, i+1] == 0:
            qpsk_symbols[j, i//2] = -1 - 1j  
        else:
            qpsk_symbols[j, i//2] = 1 - 1j
        
# Filtrage (identité pour commencer) pour chaque signal
H = 1 
transmitted_signals = np.array([np.convolve(qpsk_symbols[j], H, mode='same') for j in range(nb_signals)])

# Ajout d'un bruit blanc gaussien pour chaque signal
SNR_dB = 10  
SNR_linear = 10**(SNR_dB / 10)
noise_power = 1 / (2 * SNR_linear)  

# Génération du bruit
noise = np.sqrt(noise_power) * (np.random.normal(size=qpsk_symbols.shape) + 1j * np.random.normal(size=qpsk_symbols.shape))

# Ajout du bruit au signal QPSK
qpsk_with_noise = qpsk_symbols + noise


# Loi approximée par histogramme pour chaque signal
plt.figure()
for j in range(nb_signals):
    plt.hist(qpsk_with_noise[j], bins=20, density=True, alpha=0.5, label=f'Signal {j+1}')

# Loi approximée par méthode des noyaux pour chaque signal
epsilon2 = 0.55
D = 100
z = np.linspace(-5, 5, 1000)
for j in range(nb_signals):
    hpz = np.zeros_like(z)
    for i in range(D):
        hpz = hpz + np.exp(-(z - qpsk_with_noise[j, i])**2 / (2 * epsilon2)) / (D * np.sqrt(2 * np.pi * epsilon2))
    plt.plot(z, hpz, linewidth=3, label=f'Signal {j+1}')


plt.legend()
plt.show()
