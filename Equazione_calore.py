import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

# Definisco le condizioni al contorno
def boundary_conditions(old_T, new_T, condition = "Dirichlet"):
    if condition.lower() == "Dirichlet".lower():
        return old_T[0], old_T[-1]
    elif condition.lower() == "Neumann".lower():
        return new_T[1], new_T[-2]
    return 0, 0

# Evoluzione di uno step
def step(old_T, k, h, dt, boundary = "Neumann"):
    N_x = len(old_T)
    new_T = np.zeros(N_x)
    for i in range(1, N_x - 1):
        new_T[i] = old_T[i] + dt * k / h**2 * (old_T[i+1] + old_T[i-1] - 2 * old_T[i])
    new_T[0], new_T[-1] = boundary_conditions(old_T, new_T, condition = boundary)
    return new_T

# Soluzione che utilizza le definizioni precedenti
def solve(N_t, Temp, k, h, dt, boundary = "Neumann"):
    for i in range(1, N_t):
        Temp[i] = step(Temp[i - 1], k, h, dt, boundary)
    return Temp

# Costanti
N_x, N_t = 200, 10000
k = 5 # m**2 / s
h = 0.5 # m
dt = 0.7 * h**2 / (2 * k) # s
T = np.zeros((N_t, N_x))
T_0 = 300
A_0 = 10
lamb = N_x * h / 2
x = np.linspace(0, (N_x - 1) * h, N_x) # m

# Condizioni iniziali
for i in range(N_x):
    T[0, i] = T_0 - A_0 * np.cos(2*np.pi*x[i]/lamb)
T = solve(N_t, T, k, h, dt, "Neumann")
M = np.zeros(N_t)
for i in range(N_t):
    M[i] = max(T[i])

# Grafici
fig, ax = plt.subplots()
for i in range(0, N_t, int(N_t/ 50)):
    ax.plot([h * i for i in range(N_x)], T[i], label = "t = %1.2f s" % (i * dt))
ax.set_xlabel("x (m)")
ax.set_ylabel("T (K)")
ax.set_title("Neumann")
ax.legend()

def exp_teo(t, tau, C):
    return A_0 * np.exp(-t/tau) + C

fig1, ax1 = plt.subplots()
tempi = np.linspace(0, (N_t - 1) * dt, N_t)
p, cov = spo.curve_fit(exp_teo, tempi, M, p0 = [1, 300])
y = exp_teo(tempi, p[0], p[1])
print("tau =", p[0],"\nC =", p[1])
ax1.plot(tempi, y)
ax1.plot(tempi, M)
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude (K)")
ax1.set_title("Amplitude decrease")

# coefficiente "teorico"
alpha = lamb**2/(4*np.pi**2*p[0])
print("alpha =", alpha)
print("errore:", int(100 *(1 - k/alpha)), "%")

plt.show()
