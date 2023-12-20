import numpy as np
import matplotlib.pyplot as plt

# Cas pour un unique signal
# Paramètres
Fs = 1000          
T = 1/Fs           
fc = 100           
Ts = 1/(2*fc)      

# Génération des bits
nb_bits = 1000     # Nombre de bits
bits = np.random.randint(2, size=nb_bits)

# Mapping QPSK
qpsk_symbols = np.zeros(nb_bits//2, dtype=complex)
for i in range(0, nb_bits, 2):
    if bits[i] == 0 and bits[i+1] == 0:
        qpsk_symbols[i//2] = 1 + 1j   
    elif bits[i] == 0 and bits[i+1] == 1:
        qpsk_symbols[i//2] = -1 + 1j  
    elif bits[i] == 1 and bits[i+1] == 0:
        qpsk_symbols[i//2] = -1 - 1j  
    else:
        qpsk_symbols[i//2] = 1 - 1j
        
# Filtrage (identité pour commencer)
H = 1 
transmitted_signal = np.convolve(qpsk_symbols, H, mode='same')

# Ajout d'un bruit blanc gaussien
SNR_dB = 10  
SNR_linear = 10**(SNR_dB / 10)
noise_power = 1 / (2 * SNR_linear)  

# Génération du bruit
noise = np.sqrt(noise_power) * (np.random.normal(size=len(transmitted_signal)) + 1j * np.random.normal(size=len(transmitted_signal)))

# Ajout du bruit au signal QPSK
qpsk_with_noise = qpsk_symbols + noise

# Affichage du signal transmis
plt.figure()
plt.scatter(np.real(qpsk_symbols), np.imag(qpsk_symbols),marker='+',label="QPSK")
plt.scatter(np.real(qpsk_with_noise), np.imag(qpsk_with_noise),marker='+',label="QPSK with noise")
plt.title('Diagramme de constellation QPSK')
plt.xlabel('Partie réelle')
plt.ylabel('Partie imaginaire')
plt.legend()
plt.grid(True)
plt.show()

# Loi approximée par histogramme
plt.figure()
plt.hist(qpsk_with_noise, bins=20, density=True, alpha=0.5)

# Loi approximée par méthode des noyaux
epsilon2 = 0.55
D = 100
z = np.linspace(-5, 5, 1000)
hpz = np.zeros_like(z)
for i in range(D):
    hpz = hpz + np.exp(-(z - qpsk_with_noise[i])**2 / (2 * epsilon2)) / (D * np.sqrt(2 * np.pi * epsilon2))
plt.plot(z, hpz, linewidth=3)

# Loi approximée par méthode des noyaux avec la moyenne et l'écart-type
hmuz = np.mean(qpsk_with_noise)
hstdz = np.std(qpsk_with_noise)
hpz2 = np.exp(-(z - hmuz)**2 / (2 * hstdz**2)) / (np.sqrt(2 * np.pi) * hstdz)
plt.plot(z, hpz2, linewidth=3)

plt.show()