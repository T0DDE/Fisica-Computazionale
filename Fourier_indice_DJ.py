from pylab import *
import matplotlib.pyplot as plt
import numpy as np

# Definisco la Trasformata
def Fourier(y, T, t):
    N = len(y)
    nu = np.array([(i - N // 2) / T for i in range(N)])
    F = np.array(
        [np.sum(y * np.exp(2j * np.pi * nu[i] * t)) for i in range(N)]
    ) * (t[1] - t[0])
    return nu, F

# Definisco la Trasformata Inversa
def Fourier_inversa(y, nu_max, nu):
    N = len(y)
    t = np.array([i / nu_max for i in range(N)])
    invF = np.array(
        [np.sum(y * np.exp(- 2j * np.pi * nu * t[i])) for i in range(N)]
    ) * (nu[1] - nu[0])
    return t, invF

# Definisco le costanti, i dati e calcolo le trasformate
T, N = 1024, 1024
t = np.linspace(1, T, N)
y = loadtxt("/Users/gabri/Desktop/University/Computazionale/Programmi/Parziale 2/dow2.txt")
nu, F = Fourier(y, T, t)
nu1, F1 =  np.fft.rfftfreq(len(t)), np.fft.rfft(y)

# Grafici delle Trasformate
fig1, ax1 = plt.subplots()
ax1.plot(nu[:N//2], np.abs(F)[:N//2], label = "Trasformata di Fourier")
ax1.set_xlabel("$\\nu$")
ax1.set_ylabel("$\mathcal{F}_\\nu(y)$")
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(nu1, np.abs(F1), label = "Trasformata Fast-Fourier")
ax2.set_xlabel("$\\nu$")
ax2.set_ylim(0, 1e6)
ax2.set_ylabel("$\mathcal{F}_\\nu(y)$")
ax2.legend()

# Cancello i termini esterni al 2 % che ci interessa
for i in range(N):
    if i not in range(int(N/2 - N * 0.01), int(N/2 + N * 0.01)):
        F[i] = 0

for i in range(int(N/2)):
    if i > 0.02 * N:
        F1[i] = 0


# calcolo le antitrasformate
new_t, new_y = Fourier_inversa(F, N / T, nu)
FFinv = np.fft.irfft(F1)

# Grafico delle antitrasformate
fig, ax = plt.subplots()
ax.plot(t, y, label = "Dati originali")
ax.plot(new_t, np.real(new_y), label = "Trasformata di Fourier inversa")
ax.plot(new_t, FFinv, label = "Trasformata Fast-Fourier inversa")
ax.set_xlabel("t")
ax.set_ylabel("y")
ax.legend()

plt.show()
