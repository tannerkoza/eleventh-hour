import numpy as np
import navsim as ns

from pathlib import Path
from datetime import datetime
from scipy.special import erfc, gammainc
from navtools.constants import SPEED_OF_LIGHT
from eleventh_hour.trajectories import prepare_trajectories

DURATION = 300
TSIM = 0.02
FE_BW = 2 * (2 / 1.023e6)
JS = np.arange(0, 45, 5, dtype=float)
INTITAL_TIME = datetime(year=2023, month=12, day=4, second=18)
TRAJECTORY = "daytona_500_sdx_1s_loop"

PROJECT_PATH = Path(__file__).parents[1]
TRAJECTORY_PATH = PROJECT_PATH / "data" / "trajectories" / TRAJECTORY
CONF_PATH = PROJECT_PATH / "conf"


def compute_H(uv: np.ndarray):
    _, nchannels = uv.shape

    ux = uv[0]
    uy = uv[1]
    uz = uv[2]

    H = np.zeros((3, nchannels))

    H[0] = ux
    H[1] = ux
    H[2] = uy
    # H[3] = uy
    # H[4] = uz
    # H[5] = uz
    # H[6] = np.ones_like(ux)
    # H[7] = np.ones_like(ux)

    return H


nperiods = int(np.ceil(DURATION / TSIM)) + 1
timeseries = np.linspace(start=0, stop=DURATION, num=nperiods)

rx_pos, _ = prepare_trajectories(
    file_path=TRAJECTORY_PATH.with_suffix(".csv"), fsim=1 / TSIM
)

conf = ns.get_configuration(configuration_path=CONF_PATH)
meas_sim = ns.get_signal_simulation(
    simulation_type="measurement",
    configuration=conf,
)

zzbs = []

for js in JS:
    # set current js
    for constellation, properties in conf.constellations.emitters.items():
        if constellation in ["gps"]:
            properties.js = js

    meas_sim.generate_truth(rx_pos=rx_pos[0])
    meas_sim.simulate()

    emitters = meas_sim.emitter_states.truth
    observables = meas_sim.observables

    snr = np.array(
        [(10 ** (ob.cn0 / 10)) / (1 / TSIM) for ob in observables[0].values()]
    )
    uv = np.array([em.uv for em in emitters[0].values()]).T

    f = np.linspace(-FE_BW / 2, FE_BW / 2, 10000)
    G = 1.023e6 * (np.sin((np.pi * f) / 1.023e6) ** 2 / (np.pi * f) ** 2)
    G = G / np.sum(G)
    rmsbw = np.sqrt(np.sum(f**2 * G))  # THIS IS SO WRONG
    H = compute_H(uv).T
    I = H.T @ (rmsbw * np.diag(snr)) @ H

    if js == 0.0:
        R = np.linalg.pinv(I)
        # R = np.eye(3)

    # I = np.eye(8)
    # R = np.linalg.pinv(I) * 100

    Q = 0.5 * erfc(np.sqrt(np.sum(snr)) / np.sqrt(2))
    gamma = gammainc(1.5, np.sum(snr / 4))

    zzb = R * 2 * Q + np.linalg.pinv(I) * gamma
    zzbs.append(zzb)

    meas_sim.clear_observables()

print()
