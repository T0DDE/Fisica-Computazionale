import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
import scipy.optimize as spo

m_argon = 39.948 # uma
sigma = 3.46 # A
D = 8
Nparticles = D**2
T = 300 # K
epsilon = 0.0103 # eV
dt = 0.1 # fs
Nsteps = 10000


def f_lenn_jones(r, epsilon, sigma):
    return 48 * epsilon * np.power(
        sigma, 12) / np.power(
        r, 13) - 24 * epsilon * np.power(
        sigma, 6) / np.power(r, 7)


def init_velocity(T, number_of_particles):
    R = 2 * np.random.rand() - 1
    return R * np.sqrt(Boltzmann * T / (
        m_argon * 1.602e-19))


def get_accelerations(posx, posy):
    acc_x = np.zeros((posx.size, posx.size))
    acc_y = np.zeros((posy.size, posy.size))
    for i in range(0, posx.size - 1):
        for j in range(i + 1, posx.size):
            r_x = posx[j] - posx[i]
            r_y = posy[j] - posy[i]
            r_mod = np.sqrt(r_x**2 + r_y**2)
            F_scalar = f_lenn_jones(r_mod, epsilon, sigma)
            Fx = F_scalar * r_x / r_mod
            Fy = F_scalar * r_y / r_mod
            acc_x[i, j] = Fx / m_argon
            acc_x[j, i] = - Fx / m_argon
            acc_y[i, j] = Fy / m_argon
            acc_y[j, i] = - Fy / m_argon
    return np.sum(acc_x, axis=0), np.sum(acc_y, axis=0)

def update_pos(x, y, vx, vy, ax, ay, dt):
    return x + vx * dt + 0.5 * ax * dt**2, y + vy * dt + 0.5 * ay * dt**2

def update_vel(vx, vy, ax, ay, a1, a2, dt):
    return vx + 0.5 * (ax + a1) * dt, vy + 0.5 * (ay + a2) * dt

def run_md(dt, number_of_steps, initial_temp, x, y):
    # prima inizializzo gli array di posizioni e velocitÃ 
    posx = np.zeros((number_of_steps, Nparticles))
    posy = np.zeros((number_of_steps, Nparticles))
    vx_old = init_velocity(initial_temp, Nparticles)
    vy_old = init_velocity(initial_temp, Nparticles)
    # calcola le prime forze con le posizioni iniziali fornite da fuori
    ax, ay = get_accelerations(x, y)
    for i in range(number_of_steps):
        # qua aggiorno le posizioni:
        x, y = update_pos(x, y, vx_old, vy_old, ax, ay, dt)
        # qua aggiorno le accelerazioni:
        a1, a2 = get_accelerations(x, y)
        # qua aggiorno le velocitÃ :
        vx, vy = update_vel(vx_old, vy_old, ax, ay, a1, a2, dt)
        vx_old, vy_old = vx, vy
        # qua sovrascrivo le nuove accelerazioni alle vecchie:
        ax, ay = a1, a2
        # qui salvo le posizioni:
        posx[i, :] = x
        posy[i, :] = y
    return posx, posy, vx, vy

# qui genero le posizioni iniziali
x = np.zeros(Nparticles)
y = np.zeros(Nparticles)

for i in range(D):
    for j in range(D):
        x[j*D+i] =+ i * sigma
        y[i+D*j] =+ j * sigma


sim_posx, sim_posy, vx, vy = run_md(dt, Nsteps, T, x, y)
v = np.sqrt(vx**2 + vy**2)


def delta_r():
    squared_shift_x= np.zeros((200, Nparticles))
    squared_shift_y= np.zeros((200, Nparticles))
    for j in range(200):
        for i in range(Nparticles-1):
            squared_shift_x[j, i] = (sim_posx[j + Nsteps - 200, i] - x[i])**2
            squared_shift_y[j, i] = (sim_posy[j + Nsteps - 200, i] - y[i])**2
            squared_shift = squared_shift_x + squared_shift_y
    return np.sum(squared_shift, axis=1) / Nparticles

# QUI CALCOLO "D" CON IL LIMITE
dr_squared_t = delta_r()
D_calcolato = dr_squared_t[-1] / (4 * Nsteps * dt)
print("D calcolato =", D_calcolato)


def retta(x, D, q):
    return D * x + q


fig1, ax1 = plt.subplots()
assex_limitato = np.linspace(Nsteps - 200, Nsteps, 200)
ax1.plot(assex_limitato, dr_squared_t, '.', markersize=3)
ax1.set_xlabel("step")
ax1.set_ylabel("DeltaR(t)")
ax1.set_title("Fit")
p, cov = spo.curve_fit(retta, assex_limitato, dr_squared_t, p0 = [1, 0])
print("D fittato =", p[0], ", termine noto q =", p[1])
ax1.plot(assex_limitato, retta(assex_limitato, p[0], p[1]), label = "Fit")


def probabilita_teorica(v, C, t):
    return  C * m_argon * (v**2 * 1.602e-19 / (Boltzmann * t)) * np.exp(
    - m_argon * v**2 * 1.602e-19 / (2 * Boltzmann * t))


fig, ax = plt.subplots()
ax.hist(v, histtype = "step", bins = 10)
ax.set_xlabel("v")
ax.set_ylabel("counts")
counts, bins = np.histogram(v, bins = 10)
bins = bins[1:] - (bins[1] - bins[0]) / 2
p, cov = spo.curve_fit(probabilita_teorica, bins, counts, p0=[10, 10])
print("C =", p[0], ", Temperatura[K] =", p[1])
x = np.linspace(bins[0], bins[-1], 1000)
y = probabilita_teorica(x, p[0], p[1])
ax.plot(x, y, label="Fit", linewidth=1)


plt.show()
