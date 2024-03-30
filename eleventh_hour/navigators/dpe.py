import numpy as np
import navsim as ns
import navtools as nt
import scipy.linalg as linalg

from itertools import product
from scipy.linalg import block_diag
from navtools.conversions import ecef2lla, enu2uvw
from navsim.error_models.clock import NavigationClock
from navtools.constants import SPEED_OF_LIGHT
from eleventh_hour.navigators import (
    DPEConfiguration,
    DPEConfiguration,
    ReceiverStates,
    ParticleStates,
)


def create_spread_grid(delta: float | list, nspheres: int | list, ntiles: int = None):
    if isinstance(nspheres, int):
        delta = [delta]
        nspheres = [nspheres]

    last_max_value = 0
    subgrids = []
    for spacing, n in zip(delta, nspheres):
        max_value = spacing * n
        subgrid = np.linspace(start=spacing, stop=max_value, num=n)

        subgrids = np.append(subgrids, subgrid + last_max_value)
        last_max_value = max_value

    grid = np.append(np.flip(-subgrids), subgrids)
    grid = np.insert(grid, int(grid.size / 2), 0.0)

    if ntiles is not None:
        grid = np.tile(grid, (ntiles, 1))

    return grid


class DirectPositioning:
    def __init__(self, conf: DPEConfiguration) -> None:
        # tuning
        self.nspheres = conf.nspheres
        self.pdelta = conf.pdelta
        self.vdelta = conf.vdelta
        self.bdelta = conf.bdelta
        self.ddelta = conf.ddelta
        self.process_noise_sigma = conf.process_noise_sigma
        self.neff_percentage = conf.neff_percentage / 100

        # properties
        if conf.rx_clock_type is None:
            self.rx_clock_properties = NavigationClock(h0=0.0, h1=0.0, h2=0.0)

        else:
            self.rx_clock_properties = ns.get_clock_allan_variance_values(
                clock_name=conf.rx_clock_type
            )

        # initial states
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

        # particles
        self.delay_bias_sigma = conf.delay_bias_sigma
        self.drift_bias_sigma = conf.drift_bias_sigma

        if conf.is_grid:
            self.delay_bias_resolution = conf.delay_bias_resolution
            self.drift_bias_resolution = conf.drift_bias_resolution

            # self.nspheres = conf.nspheres
            # self.nparticles = 2 * self.nspheres * self.rx_state.size + 1
            self.__init_grid_particles()
        else:
            self.nparticles = conf.nparticles
            self.__init_rng_particles()

        print(f"Number of Particles: {self.nparticles}")

        self.weights = (1 / self.nparticles) * np.ones(self.nparticles)

        # logging
        self.__rx_states = []
        self.__covariances = []
        self.__particles = []
        self.__weights = []

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
    def particles(self):
        log = np.array(self.__particles)
        weights = np.array(self.__weights)

        particle_states = ParticleStates(
            pos=log[:, :6:2],
            vel=log[:, 1:7:2],
            clock_bias=log[:, 6],
            clock_drift=log[:, 7],
            weights=weights,
        )

        return particle_states

    @property
    def covariances(self):
        covariances = np.array(self.__covariances)

        return covariances

    # public
    def time_update(self, T: float):
        self.T = T

        # state transition
        self.__compute_A()
        self.__compute_Q()

        process_noise = np.random.multivariate_normal(
            mean=np.zeros_like(self.rx_state), cov=self.Q, size=self.nparticles
        ).T
        self.epoch_particles = self.A @ self.epoch_particles + process_noise

    def predict_particle_observables(self, emitter_states: dict):
        pranges, prange_rates = self.__predict_observables(
            emitter_states=emitter_states,
            rx_pos=self.epoch_particles[:6:2],
            rx_vel=self.epoch_particles[1:7:2],
            rx_clock_bias=self.epoch_particles[6],
            rx_clock_drift=self.epoch_particles[7],
        )

        return pranges, prange_rates

    def predict_estimate_observables(self, emitter_states: dict):
        pranges, prange_rates = self.__predict_observables(
            emitter_states=emitter_states,
            rx_pos=self.rx_state[:6:2],
            rx_vel=self.rx_state[1:7:2],
            rx_clock_bias=self.rx_state[6],
            rx_clock_drift=self.rx_state[7],
        )

        return pranges, prange_rates

    def estimate_state(self, inphase: np.ndarray, quadrature: np.ndarray):
        systems = np.unique(self.channel_systems)

        sys_powers = np.ones(self.nparticles)
        for system in systems:
            sys_indices = (self.channel_systems == system).nonzero()

            sys_inphase = inphase.T[sys_indices]
            sys_quadrature = quadrature.T[sys_indices]

            ipower = np.sum(sys_inphase, axis=0) ** 2
            qpower = np.sum(sys_quadrature, axis=0) ** 2
            sys_power = ipower + qpower

            sys_powers *= sys_power

        weights = self.weights * sys_powers

        self.weights = weights / np.sum(weights)
        self.rx_state = np.sum(self.weights * self.epoch_particles, axis=-1)
        self.log_particles()

        self.covariance = np.cov(self.epoch_particles, aweights=self.weights)

        neff = 1 / np.sum(self.weights**2)
        if neff < self.neff_percentage * self.nparticles:
            self.__resample_particles()

    def log_rx_state(self):
        self.__rx_states.append(self.rx_state.copy())

    def log_covariance(self):
        self.__covariances.append(self.covariance.copy())

    def log_particles(self):
        self.__particles.append(self.epoch_particles.copy())
        self.__weights.append(self.weights.copy())

    # private
    def __init_grid_particles(self):
        ndelay_spheres = int(3 * self.delay_bias_sigma / self.delay_bias_resolution)
        ndrift_spheres = int(3 * self.drift_bias_sigma / self.drift_bias_resolution)
        self.nparticles = 2 * max(ndelay_spheres, ndrift_spheres) + 1

        delay_deltas = create_spread_grid(
            delta=self.delay_bias_resolution, nspheres=ndelay_spheres
        )
        drift_deltas = create_spread_grid(
            delta=self.drift_bias_resolution, nspheres=ndrift_spheres
        )

        delay_points = np.linspace(0, 1, delay_deltas.size)
        drift_points = np.linspace(0, 1, drift_deltas.size)
        if delay_deltas.size > drift_deltas.size:
            drift_deltas = np.interp(delay_points, drift_points, drift_deltas)
        else:
            delay_deltas = np.interp(drift_points, delay_points, delay_deltas)

        # / 2 to ensure norm of states == delay/drift deltas
        pb_deltas = delay_deltas / 2
        vd_deltas = drift_deltas / 2

        self.deltas = np.tile(np.array([pb_deltas, vd_deltas]), (4, 1)).T
        self.epoch_particles = np.transpose(self.rx_state + self.deltas)

    def __init_rng_particles(self):
        pb_sigma = (self.delay_bias_sigma) / 2
        vd_sigma = (self.drift_bias_sigma) / 2

        P = np.diag(
            [
                pb_sigma,
                vd_sigma,
                pb_sigma,
                vd_sigma,
                pb_sigma,
                vd_sigma,
                pb_sigma,
                vd_sigma,
            ]
        )

        self.epoch_particles = np.random.multivariate_normal(
            mean=self.rx_state, cov=P, size=self.nparticles
        ).T

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
        sf = self.rx_clock_properties.h0 / 2
        sg = self.rx_clock_properties.h2 * 2 * np.pi**2

        clock_Q = SPEED_OF_LIGHT**2 * np.array(
            [
                [sf * self.T + (1 / 3) * sg * self.T**3, (1 / 2) * sg * self.T**2],
                [(1 / 2) * sg * self.T**2, sg * self.T],
            ]
        )

        self.Q = linalg.block_diag(xyz_Q, xyz_Q, xyz_Q, clock_Q)

    def __resample_particles(self):
        # multinomial
        q = np.broadcast_to(np.cumsum(self.weights), (self.nparticles, self.nparticles))
        u = np.random.uniform(0, 1, size=self.nparticles)

        indices = np.argmax(q >= u[..., np.newaxis], axis=1)

        self.epoch_particles = self.epoch_particles[:, indices]
        self.weights = np.ones_like(self.weights) / self.nparticles

    def __predict_observables(
        self,
        emitter_states: dict,
        rx_pos: np.ndarray,
        rx_vel: np.ndarray,
        rx_clock_bias: np.ndarray,
        rx_clock_drift: np.ndarray,
    ):
        # extract current state
        constellations = []
        ranges = []
        range_rates = []

        # geometric range & range rate determination
        for emitter_state in emitter_states.values():
            range, unit_vector = nt.compute_range_and_unit_vector(
                rx_pos=rx_pos, emitter_pos=emitter_state.pos
            )
            range_rate = nt.compute_range_rate(
                rx_vel=rx_vel,
                emitter_vel=emitter_state.vel,
                unit_vector=unit_vector,
            )

            constellations.append(emitter_state.constellation)
            ranges.append(range)
            range_rates.append(range_rate)

        # observable prediction
        # TODO: add group delay/drift from emitter states
        pranges = np.array(ranges) + rx_clock_bias
        prange_rates = np.array(range_rates) + rx_clock_drift

        self.channel_systems = np.array(constellations)

        return pranges, prange_rates
