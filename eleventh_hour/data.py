import numpy as np
import navsim as ns
import pandas as pd

from numpy import sin, cos
from dataclasses import dataclass
from eleventh_hour.navigators import ReceiverStates
from navtools.conversions import ecef2lla, ecef2enu, uvw2enu


@dataclass
class States:
    time: np.ndarray
    truth_lla: np.ndarray
    truth_enu_pos: np.ndarray
    truth_enu_vel: np.ndarray
    truth_clock_bias: np.ndarray
    truth_clock_drift: np.ndarray

    lla: np.ndarray
    enu_pos: np.ndarray
    enu_vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


@dataclass
class Covariances:
    time: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


@dataclass
class Errors:
    time: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


def process_truth_states(truth: ns.ReceiverTruthStates):
    lla = np.array(
        ecef2lla(
            x=truth.pos[:, 0],
            y=truth.pos[:, 1],
            z=truth.pos[:, 2],
        )
    )
    enu_pos = np.array(
        ecef2enu(
            x=truth.pos[:, 0],
            y=truth.pos[:, 1],
            z=truth.pos[:, 2],
            lat0=lla[0, 0],
            lon0=lla[1, 0],
            alt0=lla[2, 0],
        )
    )
    enu_vel = np.array(
        uvw2enu(
            u=truth.vel[:, 0],
            v=truth.vel[:, 1],
            w=truth.vel[:, 2],
            lat0=lla[0, 0],
            lon0=lla[1, 0],
        )
    )

    return lla, enu_pos, enu_vel


def process_states(
    truth: ns.ReceiverTruthStates,
    rx_states: ReceiverStates,
):
    truth_lla, truth_enu_pos, truth_enu_vel = process_truth_states(truth=truth)

    pos = rx_states.pos
    vel = rx_states.vel

    lla = np.array(
        ecef2lla(
            x=pos[0],
            y=pos[1],
            z=pos[2],
        )
    )
    enu_pos = np.array(
        ecef2enu(
            x=pos[0],
            y=pos[1],
            z=pos[2],
            lat0=truth_lla[0, 0],
            lon0=truth_lla[1, 0],
            alt0=truth_lla[2, 0],
        )
    )

    enu_vel = np.array(
        uvw2enu(
            u=vel[0],
            v=vel[1],
            w=vel[2],
            lat0=truth_lla[0, 0],
            lon0=truth_lla[1, 0],
        )
    )

    clock_bias = rx_states.clock_bias
    clock_drift = rx_states.clock_drift

    pos_error = truth_enu_pos - enu_pos
    vel_error = truth_enu_vel - enu_vel
    clock_bias_error = truth.clock_bias - clock_bias
    clock_drift_error = truth.clock_drift - clock_drift

    states = States(
        time=truth.time,
        truth_lla=truth_lla,
        truth_enu_pos=truth_enu_pos,
        truth_enu_vel=truth_enu_vel,
        truth_clock_bias=truth.clock_bias,
        truth_clock_drift=truth.clock_drift,
        lla=lla,
        enu_pos=enu_pos,
        enu_vel=enu_vel,
        clock_bias=clock_bias,
        clock_drift=clock_drift,
    )

    errors = Errors(
        time=truth.time,
        pos=pos_error,
        vel=vel_error,
        clock_bias=clock_bias_error,
        clock_drift=clock_drift_error,
    )

    return states, errors


def process_covariances(
    time: np.ndarray,
    cov: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    deg: bool = True,
):
    if deg:
        lat = np.radians(lat)
        lon = np.radians(lon)

    enu_pos_cov = []
    enu_vel_cov = []

    for epoch, (phi, lam) in enumerate(zip(lat, lon)):
        pos_cov = cov[epoch, 0:5:2, 0:5:2]
        vel_cov = cov[epoch, 1:6:2, 1:6:2]

        R = np.array(
            [
                [-sin(lam), cos(lam), np.zeros_like(lam)],
                [-cos(lam) * sin(phi), -sin(lam) * sin(phi), cos(phi)],
                [cos(lam) * cos(phi), sin(lam) * cos(phi), sin(phi)],
            ]
        )

        enu_pos_cov.append(np.diag(R @ pos_cov @ R.T))
        enu_vel_cov.append(np.diag(R @ vel_cov @ R.T))

    enu_pos_cov = np.array(enu_pos_cov).T
    enu_vel_cov = np.array(enu_vel_cov).T
    clock_bias_var = cov[:, 6, 6]
    clock_drift_var = cov[:, 7, 7]

    covariances = Covariances(
        time=time,
        pos=enu_pos_cov,
        vel=enu_vel_cov,
        clock_bias=clock_bias_var,
        clock_drift=clock_drift_var,
    )

    return covariances


def create_padded_df(data: dict):
    return pd.DataFrame({key: pd.Series(value) for key, value in data.items()})
