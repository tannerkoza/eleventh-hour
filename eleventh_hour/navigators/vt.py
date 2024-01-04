import numpy as np
import navsim as ns
import navtools as nt
import scipy.linalg as linalg

from numba import njit
from dataclasses import dataclass
from navtools.constants import SPEED_OF_LIGHT
from navsim.error_models.clock import NavigationClock


@njit(cache=True)
def compute_prange_residual_var(cn0: np.ndarray, T: float, chip_length: float):
    cn0 = 10 ** (cn0 / 10)
    return chip_length**2 * (1 / (2 * T**2 * cn0**2) + 1 / (4 * T * cn0))


@njit(cache=True)
def compute_prange_rate_residual_var(cn0: np.ndarray, T: float, wavelength: float):
    cn0 = 10 ** (cn0 / 10)
    return (wavelength / (np.pi * T)) ** 2 * (
        2 / ((T) ** 2 * cn0**2) + 2 / ((T) * cn0)
    )


@dataclass(frozen=True)
class VDFLLConfiguration:
    # tuning
    process_noise_sigma: float
    tap_spacing: float
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
class VDFLLCorrelators:
    ip: np.ndarray
    qp: np.ndarray
    ie: np.ndarray
    qe: np.ndarray
    il: np.ndarray
    ql: np.ndarray
    sub_ip: np.ndarray
    sub_qp: np.ndarray


class VDFLL:
    def __init__(self, conf: VDFLLConfiguration) -> None:
        # tuning
        self.process_noise_sigma = conf.process_noise_sigma
        self.tap_spacing = conf.tap_spacing

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
        self.__code_error_log = []
        self.__ferror_log = []
        self.__prange_error_log = []
        self.__prange_rate_error_log = []
        self.__correlator_log = []

    # properties
    @property
    def rx_states(self):
        return np.array(self.__rx_states_log)

    @property
    def code_errors(self):
        code_errors = self.__pad_log(log=self.__code_error_log)

        return code_errors

    @property
    def ferrors(self):
        ferrors = self.__pad_log(log=self.__ferror_log)

        return ferrors

    @property
    def prange_errors(self):
        prange_errors = self.__pad_log(log=self.__prange_error_log)

        return prange_errors

    @property
    def prange_rate_errors(self):
        prange_rate_errors = self.__pad_log(log=self.__prange_rate_error_log)

        return prange_rate_errors

    @property
    def correlators(self):
        return np.array(self.__correlator_log)

    # public
    def time_update(self, T: float):
        self.T = T

        self.__compute_A()
        self.__compute_Q()

        self.rx_state = self.A @ self.rx_state
        self.P = self.A @ self.P @ self.A.T + self.Q

    def loop_closure(self):
        I = np.eye(self.rx_state.size)
        self.__compute_H()
        self.__compute_R()

        if self.__is_covariance_not_ss:
            self.__compute_ss_covariance()
            self.__is_covariance_not_ss = False

        code_error, ferror = self.__discriminate()
        prange_error = code_error * self.chip_length
        prange_rate_error = ferror * -self.wavelength
        z = np.append(prange_error, prange_rate_error)

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.rx_state += K @ z
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        # logging
        self.__rx_states_log.append(self.rx_state)
        self.__code_error_log.append(code_error)
        self.__ferror_log.append(ferror)
        self.__prange_error_log.append(prange_error)
        self.__prange_rate_error_log.append(prange_rate_error)

    def update_correlator_buffers(
        self,
        prompt: ns.CorrelatorOutputs,
        early: ns.CorrelatorOutputs,
        late: ns.CorrelatorOutputs,
    ):
        # TODO: adapt this method to handle len(buff) > 1
        self.ip = prompt.inphase
        self.qp = prompt.quadrature
        self.ie = early.inphase
        self.qe = early.quadrature
        self.il = late.inphase
        self.ql = late.quadrature
        self.sub_ip = prompt.subinphase
        self.sub_qp = prompt.subquadrature

    def predict_observables(self, emitter_states: dict):
        rx_pos = self.rx_state[:6:2]
        rx_vel = self.rx_state[1:7:2]
        rx_clock_bias = self.rx_state[6]
        rx_clock_drift = self.rx_state[7]

        constellations = []
        ranges = []
        unit_vectors = []
        range_rates = []

        for emitter_state in emitter_states.values():
            range, unit_vector = nt.compute_range_and_unit_vector(
                rx_pos=rx_pos, emitter_pos=emitter_state.pos
            )
            range_rate = nt.compute_range_rate(
                rx_vel=rx_vel, emitter_vel=emitter_state.vel, unit_vector=unit_vector
            )

            constellations.append(emitter_state.constellation)
            ranges.append(range)
            unit_vectors.append(unit_vector)
            range_rates.append(range_rate)

        # TODO: add group delay from emitter states
        pranges = np.array(ranges) + rx_clock_bias
        prange_rates = np.array(range_rates) + rx_clock_drift

        self.nchannels = len(constellations)
        self.unit_vectors = np.array(unit_vectors).T
        self.__update_cycle_lengths(
            constellations=constellations, prange_rates=prange_rates
        )

        return pranges, prange_rates

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
        ux = self.unit_vectors[0]
        uy = self.unit_vectors[1]
        uz = self.unit_vectors[2]

        range_states_H = np.zeros((self.rx_state.size, self.nchannels))
        rate_states_H = np.zeros((self.rx_state.size, self.nchannels))

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

    def __discriminate(self):
        # code discriminator
        e_envelope = np.sqrt(self.ie**2 + self.qe**2)
        l_envelope = np.sqrt(self.il**2 + self.ql**2)
        code_error = (
            (1 - self.tap_spacing)
            * (e_envelope - l_envelope)
            / (e_envelope + l_envelope)
        )

        # subcorrelators
        split_ip = np.array_split(self.sub_ip, 2)
        split_qp = np.array_split(self.sub_qp, 2)

        first_ip = np.mean(split_ip[0], axis=0)
        first_qp = np.mean(split_qp[0], axis=0)
        last_ip = np.mean(split_ip[1], axis=0)
        last_qp = np.mean(split_qp[1], axis=0)

        # frequency discriminator
        cross = first_ip * last_qp - last_ip * first_qp
        dot = first_ip * last_ip + first_qp * last_qp
        ferror = np.arctan2(cross, dot) / (np.pi * self.T)

        # logging
        correlators = VDFLLCorrelators(
            ip=self.ip,
            qp=self.qp,
            ie=self.ie,
            qe=self.qe,
            il=self.il,
            ql=self.ql,
            sub_ip=self.sub_ip,
            sub_qp=self.sub_qp,
        )
        self.__correlator_log.append(correlators)

        return code_error, ferror

    def __update_cycle_lengths(self, constellations: list, prange_rates: np.ndarray):
        chip_length = []
        wavelength = []

        for constellation, prange_rate in zip(constellations, prange_rates):
            properties = self.__signal_properties.get(constellation.casefold())

            fcarrier = properties.fcarrier
            fchip = properties.fchip_data  # ! assumes tracking data channel

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

    def __pad_log(self, log: list):
        epochs = len(log)
        max_nemitters = len(max(log, key=len))
        padded_log = np.full((epochs, max_nemitters), np.nan)

        for epoch, emitters in enumerate(log):
            padded_log[epoch][0 : len(emitters)] = emitters

        return padded_log
