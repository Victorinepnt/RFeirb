import numpy as np

import matplotlib.pyplot as plt

plt.close()

# Charger le fichier npy
data = np.load('./icassp2024rfchallenge/TestSet1Mixture/TestSet1Mixture_testmixture_QPSK_CommSignal5G1.npy')
metadata = np.load('./icassp2024rfchallenge/TestSet1Mixture/TestSet1Mixture_testmixture_QPSK_CommSignal5G1_metadata.npy')

# Récupération du signal

nb_tot = len(data)
nb_signaux = 2;
#indices_signaux = np.linspace(0,nb_tot-1,nb_signaux).astype(int); #[0, 1099]
indices_signaux = np.random.choice(np.arange(nb_tot), size=nb_signaux, replace=False)
Signal = [];
mean_sig = [];


for indice in indices_signaux:
    # mean_sig.append(np.mean(np.real(data[indice])))
    # Signal.append(data[indice] + np.mean(data[indice]) * 1000);
    signal = data[indice] 
    #print("mean" + str(np.mean(data[indice])) + "mean real" + str(np.mean(np.real(data[indice]))))
    #energy = np.sum(np.abs(signal)**2)  
    normalized_signal = signal / np.mean(signal)  # normalisation
    Signal.append(normalized_signal)
    mean_sig.append(np.mean(np.real(normalized_signal)))
    

# Affichage du signal
# plt.figure()
# plt.plot(Signal_1)

# Histogramme
plt.figure()
for j in range(nb_signaux):
    plt.hist(np.real(Signal[j]), bins=20, density=True, alpha=0.5, label=f'Signal {j+1}')
    plt.legend()

# plt.figure()
# for j in range(nb_signals):
#     plt.hist(np.real(Signal[j]), bins=20, density=True, alpha=0.5, label="Partie réelle")

# plt.show()

# plt.figure()
# for j in range(nb_signals):
#     plt.hist(np.imag(Signal[j]), bins=20, density=True, alpha=0.5, label="Partie imaginaire")

# plt.show()

long = len(Signal[0]);
Z = np.zeros(long)
B = np.random.randint(0, nb_signaux, size=long)
for i in range(nb_signaux):
    Z[B == i] = Signal[i][B == i];
    print(Z[B == i])

# Z[B==0] = Signal[0][B==0];
# Z[B==1] = Signal[1][B==1];


plt.figure()
plt.hist(Z, bins=200, density = True, alpha = 0.5);

# Loi approximée par méthode des noyaux
epsilon2 = 0.01

z = np.linspace(-300, 300, 300);
hpz = np.zeros_like(z)

for i in range(long):
    hpz = hpz + np.exp(-(z - Z[i])**2 / (2 * epsilon2)) / (len(Z) * np.sqrt(2 * np.pi * epsilon2))
plt.figure()
plt.hist(Z, bins=200, density=True, alpha=0.5, cumulative=False, histtype='stepfilled',label= "histogram")
plt.plot(z, hpz, linewidth=1, label = "methode des noyaux")

# Loi approximée par méthode des noyaux avec la moyenne et l'écart-type
hmuz = np.mean(Z)
hstdz = np.std(Z)
hpz2 = np.exp(-(z - hmuz)**2 / (2 * hstdz**2)) / (np.sqrt(2 * np.pi) * hstdz)
plt.plot(z, hpz2, linewidth=1, label = "méthode moyenne/ecart type")
plt.legend()


# # Affichage du signal transmis
# plt.figure()
# plt.scatter(np.real(Signal_1), np.imag(Signal_1),marker='+',label="QPSK")
# plt.scatter(np.real(Signal_1), np.imag(Signal_1),marker='+',label="QPSK with noise")
# plt.title('Diagramme de constellation QPSK')
# plt.xlabel('Partie réelle')
# plt.ylabel('Partie imaginaire')
# plt.legend()
# plt.grid(True)
# plt.show()

