import numpy as np
import navsim as ns
import navtools as nt
import scipy.linalg as linalg

from numba import njit
from dataclasses import dataclass
from collections import defaultdict
from navtools.constants import SPEED_OF_LIGHT
from navsim.error_models.clock import NavigationClock


@njit(cache=True)
def compute_prange_residual_var(cn0: np.ndarray, T: float, chip_length: float):
    cn0 = 10 ** (cn0 / 10)
    var = chip_length**2 * (1 / (2 * T**2 * cn0**2) + 1 / (4 * T * cn0))

    return var


@njit(cache=True)
def compute_prange_rate_residual_var(cn0: np.ndarray, T: float, wavelength: float):
    cn0 = 10 ** (cn0 / 10)
    var = (wavelength / (np.pi * T)) ** 2 * (
        2 / ((T) ** 2 * cn0**2) + 2 / ((T) * cn0)
    )

    return var


@dataclass(frozen=True)
class VDFLLConfiguration:
    # tuning
    process_noise_sigma: float
    tap_spacing: float
    norm_innovation_thresh: float
    correlator_buff_size: int

    # initial states
    rx_pos: np.ndarray
    rx_vel: np.ndarray
    rx_clock_bias: float
    rx_clock_drift: float
    cn0: np.ndarray

    # properties
    rx_clock_type: str
    signal_properties: ns.SatelliteSignal


@dataclass(frozen=True)
class VDFLLReceiverStates:
    pos: np.ndarray
    vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


@dataclass(frozen=True)
class VDFLLChannelErrors:
    chip: defaultdict
    frequency: defaultdict
    prange: defaultdict
    prange_rate: defaultdict


@dataclass(frozen=True)
class VDFLLCorrelators:
    ip: defaultdict
    qp: defaultdict
    ie: defaultdict
    qe: defaultdict
    il: defaultdict
    ql: defaultdict
    sub_ip: defaultdict
    sub_qp: defaultdict


class VDFLL:
    def __init__(self, conf: VDFLLConfiguration) -> None:
        # tuning
        self.process_noise_sigma = conf.process_noise_sigma
        self.tap_spacing = conf.tap_spacing
        self.norm_innovation_thresh = conf.norm_innovation_thresh

        # initial states
        self.rx_state = np.array(
            [
                conf.rx_pos[0],
                conf.rx_vel[0],
                conf.rx_pos[1],
                conf.rx_vel[1],
                conf.rx_pos[2],
                conf.rx_vel[2],
                conf.rx_clock_bias,
                conf.rx_clock_drift,
            ]
        )
        self.cn0 = conf.cn0

        # properties
        if conf.rx_clock_type is not None:
            self.rx_clock_params = ns.get_clock_allan_variance_values(
                clock_name=conf.rx_clock_type
            )
        else:
            self.rx_clock_params = NavigationClock(h0=0.0, h1=0.0, h2=0.0)
        self.__signal_properties = conf.signal_properties

        # state
        self.P = np.eye(self.rx_state.size)
        self.__is_covariance_not_ss = True
        self.correlator_buff_size = conf.correlator_buff_size
        self.nloop_closures = 0

        # logging
        self.__rx_states_log = []

        self.__chip_error_log = defaultdict(lambda: [])
        self.__ferror_log = defaultdict(lambda: [])
        self.__prange_error_log = defaultdict(lambda: [])
        self.__prange_rate_error_log = defaultdict(lambda: [])

        self.__ip_log = defaultdict(lambda: [])
        self.__qp_log = defaultdict(lambda: [])
        self.__sub_ip_log = defaultdict(lambda: [])
        self.__sub_qp_log = defaultdict(lambda: [])

        self.__ie_log = defaultdict(lambda: [])
        self.__qe_log = defaultdict(lambda: [])

        self.__il_log = defaultdict(lambda: [])
        self.__ql_log = defaultdict(lambda: [])

    # properties
    @property
    def rx_states(self):
        rx_states_log = np.array(self.__rx_states_log)
        rx_states = VDFLLReceiverStates(
            pos=rx_states_log[:, :6:2],
            vel=rx_states_log[:, 1:7:2],
            clock_bias=rx_states_log[:, 6],
            clock_drift=rx_states_log[:, 7],
        )

        return rx_states

    @property
    def channel_errors(self):
        channel_errors = VDFLLChannelErrors(
            chip=self.__chip_error_log,
            frequency=self.__ferror_log,
            prange=self.__prange_error_log,
            prange_rate=self.__prange_rate_error_log,
        )

        return channel_errors

    @property
    def correlators(self):
        correlators = VDFLLCorrelators(
            ip=self.__ip_log,
            qp=self.__qp_log,
            ie=self.__ie_log,
            qe=self.__qe_log,
            il=self.__il_log,
            ql=self.__ql_log,
            sub_ip=self.__sub_ip_log,
            sub_qp=self.__sub_qp_log,
        )
        return correlators

    # public
    def time_update(self, T: float):
        self.T = T

        # state transition & process covariance
        self.__compute_A()
        self.__compute_Q()

        self.rx_state = self.A @ self.rx_state
        self.P = self.A @ self.P @ self.A.T + self.Q

    def predict_observables(self, emitter_states: dict):
        # extract current state
        rx_pos = self.rx_state[:6:2]
        rx_vel = self.rx_state[1:7:2]
        rx_clock_bias = self.rx_state[6]
        rx_clock_drift = self.rx_state[7]

        constellations = []
        ranges = []
        unit_vectors = []
        range_rates = []

        # geometric range & range rate determination
        for emitter_state in emitter_states.values():
            grange, unit_vector = nt.compute_range_and_unit_vector(
                rx_pos=rx_pos, emitter_pos=emitter_state.pos
            )
            grange_rate = nt.compute_range_rate(
                rx_vel=rx_vel, emitter_vel=emitter_state.vel, unit_vector=unit_vector
            )

            constellations.append(emitter_state.constellation)
            ranges.append(grange)
            unit_vectors.append(unit_vector)
            range_rates.append(grange_rate)

        # observable prediction
        # TODO: add group delay/drift from emitter states
        pranges = np.array(ranges) + rx_clock_bias
        prange_rates = np.array(range_rates) + rx_clock_drift

        # update state
        self.__update_cycle_lengths(
            constellations=constellations, prange_rates=prange_rates
        )
        self.__nchannels = len(constellations)
        self.__emitter_states = emitter_states
        self.__unit_vectors = np.array(unit_vectors).T

        return pranges, prange_rates

    def update_correlator_buffers(
        self,
        prompt: ns.CorrelatorOutputs,
        early: ns.CorrelatorOutputs,
        late: ns.CorrelatorOutputs,
    ):
        # TODO: adapt this method to handle len(buff) > 1
        self.__ip = prompt.inphase
        self.__qp = prompt.quadrature
        self.__sub_ip = prompt.subinphase
        self.__sub_qp = prompt.subquadrature
        self.__log_by_emitter(data=self.__ip, log=self.__ip_log)
        self.__log_by_emitter(data=self.__qp, log=self.__qp_log)
        self.__log_by_emitter(data=self.__sub_ip.T, log=self.__sub_ip_log)
        self.__log_by_emitter(data=self.__sub_qp.T, log=self.__sub_qp_log)

        self.__ie = early.inphase
        self.__qe = early.quadrature
        self.__log_by_emitter(data=self.__ie, log=self.__ie_log)
        self.__log_by_emitter(data=self.__qe, log=self.__qe_log)

        self.__il = late.inphase
        self.__ql = late.quadrature
        self.__log_by_emitter(data=self.__il, log=self.__il_log)
        self.__log_by_emitter(data=self.__ql, log=self.__ql_log)

    def loop_closure(self):
        # observation & measurement covariance
        self.__compute_H()
        self.__compute_R()
        I = np.eye(self.rx_state.size)

        if self.__is_covariance_not_ss:
            self.__compute_ss_covariance()  # allows for quick convergence
            self.__is_covariance_not_ss = False

        # obtain measurements
        chip_error, ferror = self.__discriminate()
        prange_error = chip_error * self.chip_length
        prange_rate_error = ferror * -self.wavelength
        z = np.append(prange_error, prange_rate_error)

        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance for fde

        norm_z = np.abs(z / np.sqrt(np.diag(S)))  # fault detection & exclusion
        if np.any(norm_z) > self.norm_innovation_thresh:
            return

        # state & covariance update
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.rx_state += K @ z
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        # logging
        self.__rx_states_log.append(self.rx_state)
        self.__log_by_emitter(data=chip_error, log=self.__chip_error_log)
        self.__log_by_emitter(data=ferror, log=self.__ferror_log)
        self.__log_by_emitter(data=prange_error, log=self.__prange_error_log)
        self.__log_by_emitter(data=prange_rate_error, log=self.__prange_rate_error_log)

    # private
    def __compute_A(self):
        xyzclock_A = np.array([[1, self.T], [0, 1]])
        self.A = linalg.block_diag(xyzclock_A, xyzclock_A, xyzclock_A, xyzclock_A)

    def __compute_Q(self):
        # position and velocity
        xyz_Q = (self.process_noise_sigma**2) * np.array(
            [
                [(1 / 3) * self.T**3, (1 / 2) * self.T**2],
                [(1 / 2) * self.T**2, self.T],
            ]
        )

        # clock
        sf = self.rx_clock_params.h0 / 2
        sg = self.rx_clock_params.h2 * 2 * np.pi**2

        clock_Q = SPEED_OF_LIGHT * np.array(
            [
                [sf * self.T + (1 / 3) * sg * self.T**3, (1 / 2) * sg * self.T**2],
                [(1 / 2) * sg * self.T**2, sg * self.T],
            ]
        )

        self.Q = linalg.block_diag(xyz_Q, xyz_Q, xyz_Q, clock_Q)

    def __compute_H(self):
        ux = self.__unit_vectors[0]
        uy = self.__unit_vectors[1]
        uz = self.__unit_vectors[2]

        range_states_H = np.zeros((self.rx_state.size, self.__nchannels))
        rate_states_H = np.zeros((self.rx_state.size, self.__nchannels))

        range_states_H[0] = ux
        range_states_H[2] = uy
        range_states_H[4] = uz
        range_states_H[6] = np.ones_like(ux)

        rate_states_H[1] = ux
        rate_states_H[3] = uy
        rate_states_H[5] = uz
        rate_states_H[7] = np.ones_like(ux)

        self.H = np.vstack((range_states_H.T, rate_states_H.T))

    def __compute_R(self):
        prange_residual_var = compute_prange_residual_var(
            cn0=self.cn0, T=self.T, chip_length=self.chip_length
        )
        prange_rate_residual_var = compute_prange_rate_residual_var(
            cn0=self.cn0, T=self.T, wavelength=self.wavelength
        )
        residual_vars = np.append(prange_residual_var, prange_rate_residual_var)

        self.R = np.diag(residual_vars)

    def __update_cycle_lengths(self, constellations: list, prange_rates: np.ndarray):
        chip_length = []
        wavelength = []

        for constellation, prange_rate in zip(constellations, prange_rates):
            properties = self.__signal_properties.get(constellation.casefold())

            fcarrier = properties.fcarrier
            fchip = properties.fchip_data  # ! assumes tracking data channel !

            fratio = fchip / fcarrier
            carrier_doppler = -prange_rate * (fcarrier / SPEED_OF_LIGHT)
            code_doppler = carrier_doppler * fratio

            chip_length.append(SPEED_OF_LIGHT / (fchip + code_doppler))
            wavelength.append(SPEED_OF_LIGHT / (fcarrier + carrier_doppler))

        self.chip_length = np.array(chip_length)
        self.wavelength = np.array(wavelength)

    def __compute_ss_covariance(self):
        DIFFERENCE_THRESHOLD = 1e-4

        I = np.eye(self.A.shape[0])
        delta_diag_P = np.diag(self.P)

        while np.any(delta_diag_P > DIFFERENCE_THRESHOLD):
            previous_P = self.P

            self.P = self.A @ self.P @ self.A.T + self.Q
            K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

            delta_diag_P = np.diag(previous_P - self.P)

    def __discriminate(self):
        # code discriminator
        e_envelope = np.sqrt(self.__ie**2 + self.__qe**2)
        l_envelope = np.sqrt(self.__il**2 + self.__ql**2)
        chip_error = (
            (1 - self.tap_spacing)
            * (e_envelope - l_envelope)
            / (e_envelope + l_envelope)
        )

        # subcorrelators
        split_ip = np.array_split(self.__sub_ip, 2)
        split_qp = np.array_split(self.__sub_qp, 2)

        first_ip = np.mean(split_ip[0], axis=0)
        first_qp = np.mean(split_qp[0], axis=0)
        last_ip = np.mean(split_ip[1], axis=0)
        last_qp = np.mean(split_qp[1], axis=0)

        # frequency discriminator
        cross = first_ip * last_qp - last_ip * first_qp
        dot = first_ip * last_ip + first_qp * last_qp
        ferror = np.arctan2(cross, dot) / (np.pi * self.T)

        return chip_error, ferror

    def __log_by_emitter(self, data: np.ndarray, log: defaultdict):
        for idx, emitter in enumerate(self.__emitter_states.values()):
            log[(emitter.constellation, emitter.id)].append(data[idx])
