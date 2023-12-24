import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import scipy.optimize as spo
import vec3d as v3d

Z1, Z2 = 2, 79
alpha_mass = 2 * spc.proton_mass + 2 * spc.neutron_mass
F0 = Z1 * Z2 * spc.e**2 / (4 * np.pi * spc.epsilon_0 * alpha_mass)
E = 5 * 1e6 * spc.electron_volt
Nparticles = 1000
d = F0 / E * alpha_mass
vel = np.sqrt(2 * E / alpha_mass)
tau = d / vel


# QUI DEFINISCO LE FUNZIONI PER VERLET E LA FORZA.
def F(r):
    mod = r.mod()
    return F0 / mod**3 * r


def step(r, v, tau):
    new_r = r + v * tau + F(r) * tau**2 / 2
    new_v = v + (F(r) + F(new_r)) * tau / 2
    return new_r, new_v


def solve(r0, v0, tau, Nsteps):
    t, r, v = [0], [r0], [v0]
    for i in range(Nsteps - 1):
        new_r, new_v = step(r[i], v[i], tau)
        t.append(t[i] + tau)
        r.append(new_r)
        v.append(new_v)
    return t, r, v


# QUI CREO LE LISTE
theta = []
theta_mezzi = []
b = []
raggiz = []
raggiy = []

# DA QUI GENERO LE PARTICELLE
for i in range(Nparticles):
    r0 = v3d.vec3d(-100 * d, (2 * np.random.rand() - 1) * d, (2 * np.random.rand() - 1) * d)
    v0 = v3d.vec3d(vel, 0, 0)
    if np.sqrt(r0.y**2 + r0.z**2) <= d:
        Nsteps = 2 * int(100 * d / d)
        t, r, v = solve(r0, v0, tau, Nsteps)
        theta.append(v0.get_angle(v[-1]))
        theta_mezzi.append(v0.get_angle(v[-1])/2)
        raggiy.append(r0.y)
        raggiz.append(r0.z)
        b.append(np.sqrt(r0.y**2 + r0.z**2))


def curva_teorica(theta, N, alpha):
    return N / (np.sin(theta/2)**alpha)


fig, ax = plt.subplots()
ax.hist(theta, histtype="step", bins=30)
counts, bins = np.histogram(theta, bins=30)
bins = bins[1:] - (bins[1] - bins[0]) / 2
p, cov = spo.curve_fit(curva_teorica, bins, counts, p0=[10e6, 4])
print(p, "\n", cov)
x = np.linspace(bins[0], bins[-1], 1000)
y = curva_teorica(x, p[0], p[1])
ax.set_yscale("log")
ax.plot(x, y, label="Fit", linewidth=1)
ax.legend()
ax.set_xlabel("$\\theta$")
ax.set_ylabel("Counts")
ax.set_title("GRAFICO 1")

fig, ax1 = plt.subplots()
ax1.plot(raggiy, raggiz, '.', markersize=1)
ax1.axis("equal")
ax1.set_xlabel("Componente y")
ax1.set_ylabel("Componente z")
ax1.set_title("GRAFICO 2")

fig, ax3 = plt.subplots()
ax3.plot(theta_mezzi, b, '.', markersize=1)
ax3.set_ylabel("b($\\theta$/2)")
ax3.set_xlabel("$\\theta$/2")
ax3.set_title("GRAFICO 3")

plt.show()
