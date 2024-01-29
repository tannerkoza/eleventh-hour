import numpy as np
import navsim as ns
import navtools as nt
import scipy.linalg as linalg

from numba import njit
from collections import defaultdict
from navtools.constants import SPEED_OF_LIGHT
from navsim.error_models.clock import NavigationClock
from navsim.simulations.correlator import CorrelatorOutputs
from eleventh_hour.navigators import (
    VDFLLConfiguration,
    ReceiverStates,
    ChannelErrors,
    Correlators,
)


class VDFLL:
    def __init__(self, conf: VDFLLConfiguration) -> None:
        # tuning
        self.proc_noise_sigma = conf.proc_noise_sigma
        self.tap_spacing = conf.tap_spacing
        self.ni_threshold = conf.ni_threshold

        # initial states
        self.cn0 = conf.cn0
        self.T = conf.T
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

        # properties
        if conf.rx_clock_type is None:
            self.rx_clock_properties = NavigationClock(h0=0.0, h1=0.0, h2=0.0)

        else:
            self.rx_clock_properties = ns.get_clock_allan_variance_values(
                clock_name=conf.rx_clock_type
            )
        self.__sig_properties = conf.sig_properties

        # cn0
        self.__ncorr_updates = 0
        self.__cn0_buffer_size = conf.cn0_buff_size
        self.__ip_cn0_buff = []
        self.__qp_cn0_buff = []

        # Kalman filter
        self.__compute_A()
        self.__compute_Q()
        self.P = np.eye(self.rx_state.size)
        self.__is_covariance_not_ss = True

        # logging
        self.__init_logging()

    def __init_logging(self):
        # state
        self.__rx_states = []
        self.__covariances = []

        # discriminator errors
        self.__chip_errors = defaultdict(lambda: [])
        self.__ferrors = defaultdict(lambda: [])
        self.__prange_errors = defaultdict(lambda: [])
        self.__prange_rate_errors = defaultdict(lambda: [])

        # correlators
        self.__ip = defaultdict(lambda: [])
        self.__qp = defaultdict(lambda: [])
        self.__ie = defaultdict(lambda: [])
        self.__qe = defaultdict(lambda: [])
        self.__il = defaultdict(lambda: [])
        self.__ql = defaultdict(lambda: [])
        self.__subip = defaultdict(lambda: [])
        self.__subqp = defaultdict(lambda: [])

        # cn0
        self.__cn0s = defaultdict(lambda: [])

    # properties
    @property
    def rx_states(self):
        log = np.array(self.__rx_states).T
        rx_states = ReceiverStates(
            pos=log[:6:2],
            vel=log[1:7:2],
            clock_bias=log[6],
            clock_drift=log[7],
        )

        return rx_states

    @property
    def covariances(self):
        covariances = np.array(self.__covariances)

        return covariances

    @property
    def channel_errors(self):
        channel_errors = ChannelErrors(
            chip=dict(self.__chip_errors),
            freq=dict(self.__ferrors),
            prange=dict(self.__prange_errors),
            prange_rate=dict(self.__prange_rate_errors),
        )

        return channel_errors

    @property
    def correlators(self):
        correlators = Correlators(
            ip=dict(self.__ip),
            qp=dict(self.__qp),
            ie=dict(self.__ie),
            qe=dict(self.__qe),
            il=dict(self.__il),
            ql=dict(self.__ql),
            subip=dict(self.__subip),
            subqp=dict(self.__subqp),
        )
        return correlators

    @property
    def cn0s(self):
        cn0s = dict(self.__cn0s)

        return cn0s

    # public
    def time_update(self, T: float):
        self.T = T

        # state transition & process covariance
        self.__compute_A()
        self.__compute_Q()

        self.rx_state = self.A @ self.rx_state
        self.P = self.A @ self.P @ self.A.T + self.Q

    def predict_observables(self, emitter_states: dict):
        epoch_emitters = list(emitter_states.keys())

        # initialize emitters attribute
        if not hasattr(self, "emitters"):
            self.emitters = epoch_emitters
        self.__update_emitters(epoch_emitters=epoch_emitters)

        # extract current state
        rx_pos = self.rx_state[:6:2]
        rx_vel = self.rx_state[1:7:2]
        rx_clock_bias = self.rx_state[6]
        rx_clock_drift = self.rx_state[7]

        constellations = []
        unit_vectors = []
        ranges = []
        range_rates = []

        # geometric range & range rate determination
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

        # observable prediction
        # TODO: add group delay/drift from emitter states
        pranges = np.array(ranges) + rx_clock_bias
        prange_rates = np.array(range_rates) + rx_clock_drift

        # update state
        self.__update_cycle_lengths(systems=constellations)
        self.__nchannels = len(constellations)
        self.__emitter_states = emitter_states
        self.__unit_vectors = np.array(unit_vectors).T

        return pranges, prange_rates

    def update_correlator_buffers(
        self,
        prompt: CorrelatorOutputs,
        early: CorrelatorOutputs,
        late: CorrelatorOutputs,
    ):
        self.ip = prompt.inphase
        self.qp = prompt.quadrature
        self.subip = prompt.subinphase
        self.subqp = prompt.subquadrature
        self.ie = early.inphase
        self.qe = early.quadrature
        self.il = late.inphase
        self.ql = late.quadrature

        self.__update_cn0_buffers(ip=prompt.inphase, qp=prompt.quadrature)

        self.__ncorr_updates += 1

        # cn0 update logic
        self.__is_cn0_updated = self.__ncorr_updates % self.__cn0_buffer_size == 0

        # logging
        self.__log_by_emitter(data=prompt.inphase, log=self.__ip)
        self.__log_by_emitter(data=prompt.quadrature, log=self.__qp)
        self.__log_by_emitter(data=early.inphase, log=self.__ie)
        self.__log_by_emitter(data=early.quadrature, log=self.__qe)
        self.__log_by_emitter(data=early.inphase, log=self.__il)
        self.__log_by_emitter(data=early.quadrature, log=self.__ql)
        self.__log_by_emitter(data=prompt.subinphase.T, log=self.__subip)
        self.__log_by_emitter(data=prompt.subquadrature.T, log=self.__subqp)

    def loop_closure(self):
        # cn0 estimation
        if self.__is_cn0_updated or self.__new_emitters:
            self.__estimate_cn0()

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

        y = np.append(prange_error, prange_rate_error)  # innovation

        # logging
        self.__log_by_emitter(data=chip_error, log=self.__chip_errors)
        self.__log_by_emitter(data=ferror, log=self.__ferrors)
        self.__log_by_emitter(data=prange_error, log=self.__prange_errors)
        self.__log_by_emitter(data=prange_rate_error, log=self.__prange_rate_errors)

        # normalized innovation filtering
        S = self.H @ self.P @ self.H.T + self.R

        norm_z = np.abs(y / np.sqrt(np.diag(S)))  # fault detection & exclusion
        if np.any(norm_z > self.ni_threshold):
            return

        # state & covariance update
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.rx_state += K @ y
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

    def log_rx_state(self):
        self.__rx_states.append(self.rx_state)

    def log_covariance(self):
        self.__covariances.append(self.P)

    def log_cn0(self):
        self.__log_by_emitter(data=self.cn0, log=self.__cn0s)

    # private
    def __compute_A(self):
        xyzclock_A = np.array([[1, self.T], [0, 1]])
        self.A = linalg.block_diag(xyzclock_A, xyzclock_A, xyzclock_A, xyzclock_A)

    def __compute_Q(self):
        # position and velocity
        xyz_Q = (self.proc_noise_sigma**2) * np.array(
            [
                [(1 / 3) * self.T**3, (1 / 2) * self.T**2],
                [(1 / 2) * self.T**2, self.T],
            ]
        )

        # clock
        sf = self.rx_clock_properties.h0 / 2
        sg = self.rx_clock_properties.h2 * 2 * np.pi**2

        clock_Q = SPEED_OF_LIGHT**2 * np.array(
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

    def __update_cycle_lengths(self, systems: list):
        chip_length = []
        wavelength = []

        for sys in systems:
            properties = self.__sig_properties.get(sys.casefold())

            fcarrier = properties.fcarrier
            fchip = properties.fchip_data  # ! assumes tracking data channel !

            chip_length.append(SPEED_OF_LIGHT / fchip)
            wavelength.append(SPEED_OF_LIGHT / fcarrier)

        self.chip_length = np.array(chip_length)
        self.wavelength = np.array(wavelength)

    def __estimate_cn0(self):
        ONE_THRESHOLD = 18
        RSCN_THRESHOLD = 22

        # pad list to account for new or removed satellites
        ip = nt.pad_list(input_list=self.__ip_cn0_buff)
        qp = nt.pad_list(input_list=self.__qp_cn0_buff)

        p = np.sqrt(ip**2 + qp**2)  # needed because not tracking phase
        p = np.ma.array(p, mask=np.isnan(p))  # mask nans to make calculations valid

        bandwidth = 1 / self.T

        # Beaulieu
        latest_buffer = p[1:, :]
        previous_buffer = p[:-1, :]

        noise_power = (np.abs(latest_buffer) - np.abs(previous_buffer)) ** 2
        data_power = 0.5 * (latest_buffer**2 + previous_buffer**2)
        snr = 1 / np.mean(noise_power / data_power, axis=0)
        cn0 = 10 * np.log10(snr * bandwidth)

        # RSCN
        if np.any(cn0 < RSCN_THRESHOLD):
            noise_power = np.mean(
                (np.abs(latest_buffer) - np.abs(previous_buffer)) ** 2, axis=0
            )
            total_power = np.mean(p**2, axis=0)

            snr = np.abs((total_power - noise_power) / noise_power)
            rscn_cn0 = 10 * np.log10(snr * bandwidth)
            cn0 = np.where(cn0 < RSCN_THRESHOLD, rscn_cn0, cn0)

        # reduces R variability poor cn0 estimation regimes
        if np.any(cn0 < ONE_THRESHOLD):
            one_cn0 = np.ones_like(cn0)
            cn0 = np.where(cn0 < ONE_THRESHOLD, one_cn0, cn0)

        self.cn0 = cn0

        # reset buffers
        self.__ip_cn0_buff = []
        self.__qp_cn0_buff = []

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
        e_envelope = np.sqrt(self.ie**2 + self.qe**2)
        l_envelope = np.sqrt(self.il**2 + self.ql**2)
        chip_error = (
            (1 - self.tap_spacing)
            * (e_envelope - l_envelope)
            / (e_envelope + l_envelope)
        )

        # subcorrelators
        split_ip = np.array_split(self.subip, 2)
        split_qp = np.array_split(self.subqp, 2)

        first_ip = np.mean(split_ip[0], axis=0)
        first_qp = np.mean(split_qp[0], axis=0)
        last_ip = np.mean(split_ip[1], axis=0) + np.random.randn(split_ip[1].size)
        last_qp = np.mean(split_qp[1], axis=0) + np.random.randn(split_ip[1].size)

        # frequency discriminator
        cross = first_ip * last_qp - last_ip * first_qp
        dot = first_ip * last_ip + first_qp * last_qp
        ferror = np.arctan2(cross, dot) / (np.pi * self.T)

        return chip_error, ferror

    def __update_emitters(self, epoch_emitters: list):
        self.__new_emitters = set(epoch_emitters) - set(self.emitters)
        self.__removed_emitters = set(self.emitters) - set(epoch_emitters)
        self.__removed_indices = [
            self.emitters.index(emitter) for emitter in self.__removed_emitters
        ]

        self.emitters = epoch_emitters

    def __update_cn0_buffers(self, ip: np.ndarray, qp: np.ndarray):
        if self.__removed_emitters:
            self.cn0 = np.delete(self.cn0, self.__removed_indices)

            ip_cn0_buff = nt.pad_list(self.__ip_cn0_buff).T
            qp_cn0_buff = nt.pad_list(self.__qp_cn0_buff).T

            new_ip_cn0_buff = np.delete(ip_cn0_buff, self.__removed_indices, axis=0).T
            new_qp_cn0_buff = np.delete(qp_cn0_buff, self.__removed_indices, axis=0).T

            self.__ip_cn0_buff = new_ip_cn0_buff.tolist()
            self.__qp_cn0_buff = new_qp_cn0_buff.tolist()

        self.__ip_cn0_buff.append(ip)
        self.__qp_cn0_buff.append(qp)

    def __log_by_emitter(self, data: np.ndarray, log: defaultdict):
        for idx, emitter in enumerate(self.__emitter_states.values()):
            log[(emitter.constellation, emitter.id)].append(data[idx])


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
