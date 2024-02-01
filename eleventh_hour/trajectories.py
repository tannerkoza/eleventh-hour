import numpy as np

from pathlib import Path
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from navtools.conversions import lla2ecef


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

    cs = CubicSpline(x=file_time, y=ecef_pos)
    rx_pos = cs(sim_time)
    rx_vel = cs(sim_time, 1)

    return rx_pos, rx_vel


# def prepare_trajectories(
#     file_path: str | Path,
#     fsim: float,
# ):
#     data = np.loadtxt(fname=file_path, delimiter=",", skiprows=1)
#     ecef_pos = data[:, :3]
#     ecef_vel = data[:, 3:6]
#     file_time = data[:, 6]
#     ypr_att = data[:, 7:10]  # ZYX

#     sim_period = 1 / fsim
#     sim_time = np.arange(0, file_time[-1] + sim_period, sim_period)

#     chs = CubicHermiteSpline(x=file_time, y=ecef_pos, dydx=ecef_vel)
#     cs = CubicSpline(x=file_time, y=ypr_att)

#     rx_pos = chs(sim_time)
#     rx_vel = chs(sim_time, 1)
#     rx_att = cs(sim_time)

#     return rx_pos, rx_vel, rx_att
