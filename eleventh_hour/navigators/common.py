from numpy import ndarray
from dataclasses import dataclass
from collections import defaultdict
from navsim import SatelliteSignal


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


@dataclass(frozen=True)
class NavigatorConfiguration:
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
class VDFLLConfiguration(NavigatorConfiguration):
    # tuning
    proc_noise_sigma: float
    tap_spacing: float
    ni_threshold: float  # normalized innovation filter threshold
    cn0_buff_size: int


@dataclass(frozen=True)
class DPEConfiguration(NavigatorConfiguration):
    # grids
    pos_spacings: list
    bias_spacings: list
    posbias_nstates: list
    vel_spacings: list
    drift_spacings: list
    veldrift_nstates: list


@dataclass(frozen=True)
class DPESIRConfiguration(NavigatorConfiguration):
    # tuning
    nspheres: int
    process_noise_sigma: float
    pdelta: float = 7.0
    vdelta: float = 2.5
    bdelta: float = 7.0
    ddelta: float = 0.5
