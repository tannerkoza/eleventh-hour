import numpy as np
import navsim as ns
import navtools as nt
import scipy.linalg as linalg

from itertools import product
from scipy.linalg import block_diag
from navsim.error_models.clock import NavigationClock
from navtools.constants import SPEED_OF_LIGHT
from navtools.conversions import ecef2lla, enu2ecef, enu2uvw
from eleventh_hour.navigators import (
    DPEConfiguration,
    DPESIRConfiguration,
    ReceiverStates,
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

    if ntiles is not None:
        grid = np.tile(grid, (ntiles, 1))

    return grid


class DirectPositioning:
    def __init__(self, conf: DPEConfiguration) -> None:
        # initial states
        self.T = conf.T
        self.rx_state = np.array(
            [
                conf.rx_pos[0],
                conf.rx_pos[1],
                conf.rx_pos[2],
                conf.rx_clock_bias,
                conf.rx_vel[0],
                conf.rx_vel[1],
                conf.rx_vel[2],
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

        # grids
        self.update_pbgrid_deltas(
            pdeltas=conf.pos_spacings,
            bdeltas=conf.bias_spacings,
            nstates=conf.posbias_nstates,
        )
        self.update_vdgrid_deltas(
            vdeltas=conf.vel_spacings,
            ddeltas=conf.drift_spacings,
            nstates=conf.veldrift_nstates,
        )

        self.__init_logging()

    @property
    def rx_states(self):
        log = np.array(self.__rx_states).T
        rx_states = ReceiverStates(
            pos=log[:3],
            vel=log[4:7],
            clock_bias=log[3],
            clock_drift=log[7],
        )

        return rx_states

    def __init_logging(self):
        # state
        self.__rx_states = []
        self.__covariances = []

    # public
    def predict_rx_observables(self, emitter_states: dict):
        new_emitters = list(emitter_states.keys())

        # initialize emitters attribute
        if not hasattr(self, "emitters"):
            self.emitters = new_emitters

        ranges, range_rates, constellations = self.__predict_observables(
            emitter_states=emitter_states,
            pos=self.rx_state[:3],
            vel=self.rx_state[4:7],
        )

        pranges = np.array(ranges) + self.rx_state[3]
        prange_rates = np.array(range_rates) + self.rx_state[7]

        self.__update_cycle_lengths(
            constellations=constellations,
            prange_rates=prange_rates,
        )

        return pranges, prange_rates

    def predict_grid_observables(self, emitter_states: dict):
        new_emitters = list(emitter_states.keys())

        # initialize emitters attribute
        if not hasattr(self, "emitters"):
            self.emitters = new_emitters

        pgrid = self.pbgrid[:, :3]
        vrx_state = np.broadcast_to(self.rx_state[4:7], pgrid.shape)

        ranges, _, _ = self.__predict_observables(
            emitter_states=emitter_states,
            pos=pgrid,
            vel=vrx_state,
        )

        vgrid = self.vdgrid[:, :3]
        prx_state = np.broadcast_to(self.rx_state[:3], vgrid.shape)

        _, range_rates, _ = self.__predict_observables(
            emitter_states=emitter_states,
            pos=prx_state,
            vel=vgrid,
        )

        pranges = np.transpose(np.array(ranges) + self.pbgrid[:, 3])
        prange_rates = np.transpose(np.array(range_rates) + self.vdgrid[:, 3])

        return pranges, prange_rates

    def filter_grids(
        self,
        true_states: np.ndarray,
        errbuff_pct: float = 0.0,
        is_filtered: bool = False,
    ):
        MIN_PBERROR = 0.5
        MIN_VDERROR = 0.25

        # set grids
        self.__set_pbgrid()
        self.__set_vdgrid()

        if is_filtered:
            true_pberror = np.abs(self.rx_state[:4] - true_states[:4])
            true_vderror = np.abs(self.rx_state[4:] - true_states[4:])

            pberror = (
                true_pberror if np.any(true_pberror > MIN_PBERROR) else MIN_PBERROR
            )
            vderror = (
                true_vderror if np.any(true_vderror > MIN_VDERROR) else MIN_VDERROR
            )

            pbgrid_errors = np.abs(self.rx_state[:4] - self.pbgrid)
            vdgrid_errors = np.abs(self.rx_state[4:] - self.vdgrid)

            errbuff_pct = errbuff_pct / 100 + 1.0
            valid_pb = np.all(pbgrid_errors <= errbuff_pct * pberror, axis=1).nonzero()
            print(f"npbstates: {len(valid_pb[0])}")
            valid_vd = np.all(vdgrid_errors <= errbuff_pct * vderror, axis=1).nonzero()
            print(f"nvdstates: {len(valid_vd[0])}")

            self.pbgrid = self.pbgrid[valid_pb]
            self.vdgrid = self.vdgrid[valid_vd]

    def estimate_pb(self, inphase: np.ndarray, quadrature: np.ndarray):
        ipower = np.sum(inphase, axis=1) ** 2
        qpower = np.sum(quadrature, axis=1) ** 2
        power = ipower + qpower

        norm_weights = power / np.sum(power)

        pb = np.sum(norm_weights * self.pbgrid.T, axis=-1)

        residuals = self.pbgrid - pb
        # self.pbcov = self.__compute_covariance(
        #     residuals=residuals, weights=norm_weights
        # )
        self.rx_state[:4] = pb

    def estimate_vd(self, inphase: np.ndarray, quadrature: np.ndarray):
        ipower = np.sum(inphase, axis=1) ** 2
        qpower = np.sum(quadrature, axis=1) ** 2
        prompt = ipower + qpower

        norm_weights = prompt / np.sum(prompt)

        vd = np.sum(norm_weights * self.vdgrid.T, axis=-1)

        residuals = self.vdgrid - vd
        # self.vdcov = self.__compute_covariance(
        #     residuals=residuals, weights=norm_weights
        # )
        self.rx_state[4:] = vd

    def update_pbgrid_deltas(self, pdeltas: list, bdeltas: list, nstates: list):
        enu_pdeltas = create_spread_grid(delta=pdeltas, nspheres=nstates, ntiles=3)
        biases = create_spread_grid(delta=bdeltas, nspheres=nstates) + self.rx_state[3]
        enubias_deltas = np.vstack([enu_pdeltas, biases])

        self.pbgrid_enu = np.array(list(product(*enubias_deltas))).T

    def update_vdgrid_deltas(self, vdeltas: list, ddeltas: list, nstates: list):
        enu_vdeltas = create_spread_grid(delta=vdeltas, nspheres=nstates, ntiles=3)
        drift_deltas = create_spread_grid(delta=ddeltas, nspheres=nstates)
        enudrift_deltas = np.vstack([enu_vdeltas, drift_deltas])

        # cartesian product of grid offsets
        self.vdgrid_enu = np.array(list(product(*enudrift_deltas))).T

    def time_update(self):
        # state transition & process covariance
        self.__compute_A()

        rx_state = np.array(
            [
                self.rx_state[0],
                self.rx_state[4],
                self.rx_state[1],
                self.rx_state[5],
                self.rx_state[2],
                self.rx_state[6],
                self.rx_state[3],
                self.rx_state[7],
            ]
        )

        rx_state = self.A @ rx_state

        self.rx_state = np.array(
            [
                rx_state[0],
                rx_state[2],
                rx_state[4],
                rx_state[6],
                rx_state[1],
                rx_state[3],
                rx_state[5],
                rx_state[7],
            ]
        )

    def log_rx_state(self):
        self.__rx_states.append(self.rx_state.copy())

    # private
    def __update_cycle_lengths(self, constellations: list, prange_rates: np.ndarray):
        chip_length = []
        wavelength = []

        for constellation, prange_rate in zip(constellations, prange_rates):
            properties = self.__sig_properties.get(constellation.casefold())

            fcarrier = properties.fcarrier
            fchip = properties.fchip_data  # ! assumes tracking data channel !

            fratio = fchip / fcarrier
            carrier_doppler = -prange_rate * (fcarrier / SPEED_OF_LIGHT)
            code_doppler = carrier_doppler * fratio

            chip_length.append(SPEED_OF_LIGHT / (fchip + code_doppler))
            wavelength.append(SPEED_OF_LIGHT / (fcarrier + carrier_doppler))

        self.chip_length = np.array(chip_length)
        self.wavelength = np.array(wavelength)

    def __predict_observables(
        self, emitter_states: dict, pos: np.ndarray, vel: np.ndarray
    ):
        ranges = []
        range_rates = []
        constellations = []

        for emitter_state in emitter_states.values():
            range, unit_vectors = nt.compute_range_and_unit_vector(
                rx_pos=pos, emitter_pos=emitter_state.pos
            )
            range_rate = nt.compute_range_rate(
                rx_vel=vel,
                emitter_vel=emitter_state.vel,
                unit_vector=unit_vectors,
            )

            ranges.append(range)
            range_rates.append(range_rate)
            constellations.append(emitter_state.constellation)

        return ranges, range_rates, constellations

    def __compute_covariance(self, residuals: np.ndarray, weights: np.ndarray):
        covs = []

        for wgt, res in zip(weights, residuals):
            cov = np.outer(res, res)
            covs.append(wgt * cov)

        cov = np.sum(covs, axis=0)

        return cov

    def __set_pbgrid(self):
        rx_lla = ecef2lla(x=self.rx_state[0], y=self.rx_state[1], z=self.rx_state[2])
        pgrid_ecef = np.array(
            enu2ecef(
                east=self.pbgrid_enu[0],
                north=self.pbgrid_enu[1],
                up=self.pbgrid_enu[2],
                lat0=rx_lla.lat,
                lon0=rx_lla.lon,
                h0=rx_lla.alt,
            )
        )

        self.pbgrid = np.vstack([pgrid_ecef, self.pbgrid_enu[3]]).T

    def __set_vdgrid(self):
        rx_lla = ecef2lla(x=self.rx_state[0], y=self.rx_state[1], z=self.rx_state[2])
        vgrid_ecef = np.array(
            enu2uvw(
                east=self.vdgrid_enu[0],
                north=self.vdgrid_enu[1],
                up=self.vdgrid_enu[2],
                lat0=rx_lla.lat,
                lon0=rx_lla.lon,
            )
        )

        self.vdgrid = (
            np.vstack([vgrid_ecef, self.vdgrid_enu[3]]) + self.rx_state[4:, np.newaxis]
        ).T

    def __compute_A(self):
        xyzclock_A = np.array([[1, self.T], [0, 1]])
        self.A = linalg.block_diag(xyzclock_A, xyzclock_A, xyzclock_A, xyzclock_A)


class DirectPositioningSIR:
    def __init__(self, conf: DPESIRConfiguration) -> None:
        # tuning
        self.nspheres = conf.nspheres
        self.pdelta = conf.pdelta
        self.vdelta = conf.vdelta
        self.bdelta = conf.bdelta
        self.ddelta = conf.ddelta
        self.process_noise_sigma = conf.process_noise_sigma

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
        self.__init_particles()

        # logging
        self.__rx_states = []
        self.__particles = []

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
    def rx_particles(self):
        log = np.array(self.__particles)
        rx_states = ReceiverStates(
            pos=log[:, :6:2],
            vel=log[:, 1:7:2],
            clock_bias=log[:, 6],
            clock_drift=log[:, 7],
        )

        return rx_states

    # public
    def time_update(self, T: float):
        self.T = T

        # state transition
        self.__compute_A()
        self.__compute_Q()

        process_noise = np.random.multivariate_normal(
            mean=np.zeros_like(self.rx_state), cov=self.Q, size=self.nparticles
        ).T
        self.particles = self.A @ self.particles + process_noise

    def predict_observables(self, emitter_states: dict):
        # extract current state
        constellations = []
        unit_vectors = []
        ranges = []
        range_rates = []

        # geometric range & range rate determination
        for emitter_state in emitter_states.values():
            range, unit_vector = nt.compute_range_and_unit_vector(
                rx_pos=self.particles[:6:2], emitter_pos=emitter_state.pos
            )
            range_rate = nt.compute_range_rate(
                rx_vel=self.particles[1:7:2],
                emitter_vel=emitter_state.vel,
                unit_vector=unit_vector,
            )

            constellations.append(emitter_state.constellation)
            ranges.append(range)
            unit_vectors.append(unit_vector)
            range_rates.append(range_rate)

        # observable prediction
        # TODO: add group delay/drift from emitter states
        pranges = np.array(ranges) + self.particles[6]
        prange_rates = np.array(range_rates) + self.particles[7]

        return pranges, prange_rates

    def estimate_state(self, inphase: np.ndarray, quadrature: np.ndarray):
        ipower = np.sum(inphase, axis=1) ** 2
        qpower = np.sum(quadrature, axis=1) ** 2
        power = ipower + qpower

        self.weights = power / np.sum(power)
        self.rx_state = np.sum(self.weights * self.particles, axis=-1)

        residuals = self.particles.T - self.rx_state

        neff = 1 / np.sum(self.weights**2)
        if neff < (2 / 3) * self.nparticles:
            self.__resample_particles()

    def log_rx_state(self):
        self.__rx_states.append(self.rx_state.copy())

    def log_particles(self):
        self.__particles.append(self.particles.copy())

    # private
    def __init_particles(self):
        self.nparticles = 2 * self.nspheres * self.rx_state.size + 1
        rx_state = np.broadcast_to(
            self.rx_state, (self.nparticles, self.rx_state.size)
        ).T

        pdeltas = create_spread_grid(delta=self.pdelta, nspheres=self.nspheres)
        vdeltas = create_spread_grid(delta=self.vdelta, nspheres=self.nspheres)
        bdeltas = create_spread_grid(delta=self.bdelta, nspheres=self.nspheres)
        ddeltas = create_spread_grid(delta=self.ddelta, nspheres=self.nspheres)

        deltas = block_diag(
            pdeltas, vdeltas, pdeltas, vdeltas, pdeltas, vdeltas, bdeltas, ddeltas
        )
        deltas = np.hstack((np.zeros_like(self.rx_state)[..., np.newaxis], deltas))

        self.particles = rx_state + deltas

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
        q = np.cumsum(self.weights)
        u = np.random.uniform(0, 1, size=self.nparticles)

        indices = np.array([np.argmax(q >= value) for value in u])

        self.particles = self.particles[:, indices]
        self.weights = np.ones_like(self.weights) / self.nparticles
