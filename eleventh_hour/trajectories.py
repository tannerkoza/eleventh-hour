import numpy as np

from pathlib import Path
from navtools.conversions import lla2ecef
from scipy.interpolate import CubicSpline, CubicHermiteSpline, BSpline


def prepare_trajectories(file_path: str | Path, fsim: float):
    data = np.loadtxt(fname=file_path, delimiter=",", skiprows=1)

    lat = data[:, 0]
    lon = data[:, 1]
    alt = data[:, 2]
    file_time = data[:, 3]

    file_T = np.mean(np.diff(file_time))
    sim_T = 1 / fsim

    sim_time = np.arange(0, file_time[-1] + sim_T, sim_T)

    ecef_pos = np.array(lla2ecef(lat=lat, lon=lon, alt=alt)).T
    two_point_ecef_vel = (
        np.diff(ecef_pos, axis=0, append=ecef_pos[-1].reshape(1, -1)) / file_T
    )
    ecef_vel = 0.5 * np.array(
        [
            sum(two_point_ecef_vel[current : current + 2])
            for current in range(0, len(two_point_ecef_vel), 1)
        ]
    )

    bs = BSpline(t=file_time, c=ecef_pos, k=2, axis=0)
    rx_pos = bs(sim_time)
    rx_vel = np.diff(rx_pos, axis=0, append=rx_pos[-1].reshape(1, -1)) / sim_T
    # rx_vel = cs(sim_time, 1)

    # chs = CubicHermiteSpline(x=file_time, y=ecef_pos, dydx=ecef_vel)

    # rx_pos = chs(sim_time)
    # rx_vel = chs(sim_time, 1)

    return rx_pos, rx_vel
