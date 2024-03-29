import numpy as np
import matplotlib.pyplot as plt

from numba import njit


@njit(cache=True)
def compute_tle_errors(sigma: float, tau: float, T: float, duration: float):
    vel_error = []
    pos_error = []

    size = duration / T

    a = np.exp(-T / tau)

    ve = 0.0
    pe = 0.0
    for _ in range(int(size)):
        vel_error.append(ve)
        pos_error.append(pe)

        ve = a * ve + sigma * np.random.randn()
        pe += ve * T

    return np.array(pos_error), np.array(vel_error)


duration = 800
sigma = 0.0005
tau = 4000
T = 0.02
sims = 1e3

for _ in range(int(sims)):
    pe1, ve1 = compute_tle_errors(sigma=sigma, tau=tau, T=T, duration=duration)
    pe2, ve2 = compute_tle_errors(sigma=sigma, tau=tau, T=T, duration=duration)
    pe3, ve3 = compute_tle_errors(sigma=sigma, tau=tau, T=T, duration=duration)

    pe = np.linalg.norm(np.vstack((pe1, pe2, pe3)), axis=0)
    ve = np.linalg.norm(np.vstack((ve1, ve2, ve3)), axis=0)

    plt.plot(pe)

# plt.plot(ve)
plt.show()

# xk = np.array(m[:-2])
# xk1 = np.array(m[1:-1])

# a = (1 / (xk.T @ xk)) * xk.T @ xk1
# print(f"a estimate: {a}")
# sigma = np.sqrt(np.mean((xk1 - a * xk) ** 2))
# print(f"sigma estimate: {sigma}")
# tau = -dt / (np.log(a))
# print(f"tau estimate: {tau}")
# plt.show()
