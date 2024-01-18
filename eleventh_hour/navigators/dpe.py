import numpy as np
import navsim as ns
import navtools as nt

from itertools import product
from collections import defaultdict
from navsim.error_models.clock import NavigationClock
from navtools.constants import SPEED_OF_LIGHT
from navtools.conversions import ecef2lla, enu2ecef, enu2uvw
from eleventh_hour.navigators import DirectPositioningConfiguration


def create_spread_grid(spacings: float | list, nstates: int | list, ntiles: int = None):
    if isinstance(nstates, int):
        spacings = [spacings]
        nstates = [nstates]

    last_max_value = 0
    subgrids = []
    for spacing, n in zip(spacings, nstates):
        max_value = spacing * n
        subgrid = np.linspace(start=spacing, stop=max_value, num=n)

        subgrids = np.append(subgrids, subgrid + last_max_value)
        last_max_value = max_value

    mirrored_grid = np.append(np.flip(-subgrids), subgrids)
    zero_index = int(mirrored_grid.size / 2)
    grid = np.insert(mirrored_grid, zero_index, 0.0)

    if ntiles is not None:
        grid = np.tile(grid, (ntiles, 1))

    return grid


class DirectPositioning:
    def __init__(self, conf: DirectPositioningConfiguration) -> None:
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
        self.update_posbias_grid(
            pos_spacings=conf.pos_spacings,
            bias_spacings=conf.bias_spacings,
            nstates=conf.posbias_nstates,
        )
        self.update_veldrift_grid(
            vel_spacings=conf.vel_spacings,
            drift_spacings=conf.drift_spacings,
            nstates=conf.veldrift_nstates,
        )

        print()

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

    # public
    def predict_observables(self, emitter_states: dict):
        new_emitters = list(emitter_states.keys())

        # initialize emitters attribute
        if not hasattr(self, "emitters"):
            self.emitters = new_emitters

        ranges = []
        range_rates = []
        constellations = []

        for emitter_state in emitter_states.values():
            range, unit_vectors = nt.compute_range_and_unit_vector(
                rx_pos=self.posbias_grid[:3], emitter_pos=emitter_state.pos
            )
            range_rate = nt.compute_range_rate(
                rx_vel=self.veldrift_grid[:3],
                emitter_vel=emitter_state.vel,
                unit_vector=unit_vectors,
            )

            ranges.append(range)
            range_rates.append(range_rate)
            constellations.append(emitter_state.constellation)

        self.pranges = np.transpose(np.array(ranges) + self.posbias_grid[3])
        self.prange_rates = np.transpose(np.array(range_rates) + self.veldrift_grid[3])

        _, nprange_rates = self.prange_rates.shape
        rx_state_idx = int(nprange_rates / 2)

        self.__update_cycle_lengths(
            constellations=constellations,
            prange_rates=self.prange_rates[:, rx_state_idx],
        )

        self.code_phases = self.pranges / self.chip_length  # TODO: mod this with nchips
        self.fdopplers = self.prange_rates / -self.wavelength

        print()

    def update_posbias_grid(
        self, pos_spacings: list, bias_spacings: list, nstates: list
    ):
        enu_offsets = create_spread_grid(
            spacings=pos_spacings, nstates=nstates, ntiles=3
        )
        bias_offsets = create_spread_grid(spacings=bias_spacings, nstates=nstates)
        enubias_offsets = np.vstack([enu_offsets, bias_offsets])

        enubias_grid = np.array(list(product(*enubias_offsets))).T

        rx_lla = ecef2lla(x=self.rx_state[0], y=self.rx_state[1], z=self.rx_state[2])
        ecef_grid = np.array(
            enu2ecef(
                east=enubias_grid[0],
                north=enubias_grid[1],
                up=enubias_grid[2],
                lat0=rx_lla.lat,
                lon0=rx_lla.lon,
                h0=rx_lla.alt,
            )
        )

        self.posbias_grid = np.vstack([ecef_grid, enubias_grid[3]])

    def update_veldrift_grid(
        self, vel_spacings: list, drift_spacings: list, nstates: list
    ):
        enu_offsets = create_spread_grid(
            spacings=vel_spacings, nstates=nstates, ntiles=3
        )
        drift_offsets = create_spread_grid(spacings=drift_spacings, nstates=nstates)
        enudrift_offsets = np.vstack([enu_offsets, drift_offsets])

        # cartesian product of grid offsets
        enudrift_grid = np.array(list(product(*enudrift_offsets))).T

        rx_lla = ecef2lla(x=self.rx_state[0], y=self.rx_state[1], z=self.rx_state[2])
        ecef_grid = np.array(
            enu2uvw(
                east=enudrift_grid[0],
                north=enudrift_grid[1],
                up=enudrift_grid[2],
                lat0=rx_lla.lat,
                lon0=rx_lla.lon,
            )
        )

        self.veldrift_grid = (
            np.vstack([ecef_grid, enudrift_grid[3]]).T + self.rx_state[4:]
        ).T

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
