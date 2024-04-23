import numpy as np
import matplotlib.pyplot as plt

from navsim.error_models.clock import HIGH_QUALITY_TCXO
from navtools.constants import SPEED_OF_LIGHT

clock = HIGH_QUALITY_TCXO
T = 0.02
nperiods = 500

sf = clock.h0 / 2
sg = clock.h2 * 2 * np.pi**2

# error covariance noise
covariance = np.array(
    [
        [sf * T + (1 / 3) * sg * T**3, (1 / 2) * sg * T**2],
        [(1 / 2) * sg * T**2, sg * T],
    ]
)  # [s], [s/s]

for _ in range(100):
    bias_noise, drift_noise = np.random.multivariate_normal(
        mean=[0, 0], cov=covariance, size=nperiods
    ).T

    drift_ss = np.cumsum(drift_noise)
    bias_s = np.cumsum(drift_ss * T) + np.cumsum(bias_noise)

    drift_ms = drift_ss * SPEED_OF_LIGHT
    bias_m = bias_s * SPEED_OF_LIGHT

    plt.plot(drift_ms, c="red")

    sf = clock.h0 / 2
    sg = clock.h2 * 2 * np.pi**2

    clock_Q = SPEED_OF_LIGHT**2 * np.array(
        [
            [sf * T + (1 / 3) * sg * T**3, (1 / 2) * sg * T**2],
            [(1 / 2) * sg * T**2, sg * T],
        ]
    )

    states = np.zeros(2)
    states2 = states
    A = np.array([[1, T], [0, 1]])

    bias = []
    bias2 = []
    drift = []

    for _ in range(nperiods):
        bias.append(states[0])
        drift.append(states[1])
        process_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=clock_Q)

        states = A @ states + process_noise

    plt.plot(drift, c="blue")
    # plt.plot(2, c="green")


plt.show()
