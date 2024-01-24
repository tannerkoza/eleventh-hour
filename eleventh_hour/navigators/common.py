import numpy as np
import pandas as pd

from numpy import ndarray
from numpy import sin, cos
from dataclasses import dataclass
from collections import defaultdict
from navsim import SatelliteSignal, ReceiverTruthStates
from navtools.conversions import ecef2lla, ecef2enu, uvw2enu


# navigator logging
@dataclass(frozen=True, repr=False)
class ReceiverStates:
    pos: ndarray
    vel: ndarray
    clock_bias: ndarray
    clock_drift: ndarray


@dataclass(frozen=True, repr=False)
class ChannelErrors:
    chip: defaultdict
    freq: defaultdict
    prange: defaultdict
    prange_rate: defaultdict


@dataclass(frozen=True, repr=False)
class Correlators:
    ip: defaultdict
    qp: defaultdict
    ie: defaultdict
    qe: defaultdict
    il: defaultdict
    ql: defaultdict
    subip: defaultdict
    subqp: defaultdict


# navigator configuration
@dataclass(frozen=True)
class VDFLLConfiguration:
    # tuning
    proc_noise_sigma: float
    tap_spacing: float
    ni_threshold: float  # normalized innovation filter threshold
    cn0_buff_size: int

    # initial states
    cn0: ndarray
    rx_pos: ndarray
    rx_vel: ndarray
    rx_clock_bias: float
    rx_clock_drift: float
    T: float  # integration period

    # properties
    rx_clock_type: str
    sig_properties: SatelliteSignal


@dataclass(frozen=True)
class DPEConfiguration:
    # tuning
    nspheres: int
    pdelta: float
    vdelta: float
    bdelta: float
    ddelta: float
    process_noise_sigma: float
    neff_percentage: float

    # initial states
    rx_pos: ndarray
    rx_vel: ndarray
    rx_clock_bias: float
    rx_clock_drift: float
    T: float  # integration period

    # properties
    rx_clock_type: str


# simulation results
@dataclass
class StateResults:
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
class CovarianceResults:
    time: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


@dataclass
class ErrorResults:
    time: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


@dataclass(frozen=True)
class SimulationResults:
    states: StateResults
    errors: ErrorResults
    covariances: CovarianceResults

    # correlator sim specific
    chip_errors: dict
    ferrors: dict
    prange_errors: dict
    prange_rate_errors: dict

    # vector procesing specific
    cn0s: dict = None
    channel_errors: ChannelErrors = None
    correlators: Correlators = None

    # dpe specific
    particles: np.ndarray = None


def process_truth_states(truth: ReceiverTruthStates):
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


def process_state_results(
    truth: ReceiverTruthStates,
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

    states = StateResults(
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

    errors = ErrorResults(
        time=truth.time,
        pos=pos_error,
        vel=vel_error,
        clock_bias=clock_bias_error,
        clock_drift=clock_drift_error,
    )

    return states, errors


def process_covariance_results(
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

    covariances = CovarianceResults(
        time=time,
        pos=enu_pos_cov,
        vel=enu_vel_cov,
        clock_bias=clock_bias_var,
        clock_drift=clock_drift_var,
    )

    return covariances


def create_padded_df(data: dict):
    return pd.DataFrame({key: pd.Series(value) for key, value in data.items()})
