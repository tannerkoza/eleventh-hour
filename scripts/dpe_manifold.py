import itertools
import numpy as np
import navsim as ns
import navtools as nt
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
import eleventh_hour.trajectories as eh

from pathlib import Path
from scipy.interpolate import bisplrep, bisplev
from navtools.conversions import ecef2lla, enu2ecef

TRAJECTORY = "flight_australia_sdx_1s_onego"

# dpe
IS_COHERENT = True
NE_DELTA = 0.001
NSPHERES = 500
NTILES = 2

# plotting
IS_SMOOTHED = False
INTERP_RESOLUTION = 0.001
CMAP = cm.rainbow
CONTEXT = "paper"
# plt.rcParams["figure.figsize"] = [20, 10]

# path literals
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
TRAJECTORY_PATH = DATA_PATH / "trajectories" / TRAJECTORY


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


def predict_observables(
    emitter_states: dict,
    rx_pos: np.ndarray,
    rx_vel: np.ndarray,
    rx_clock_bias: np.ndarray,
    rx_clock_drift: np.ndarray,
):
    # extract current state
    channel_systems = []
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

        channel_systems.append(emitter_state.constellation)
        ranges.append(range)
        range_rates.append(range_rate)

    # observable prediction
    pranges = np.array(ranges) + rx_clock_bias
    prange_rates = np.array(range_rates) + rx_clock_drift

    return np.array(channel_systems), pranges, prange_rates


def compute_manifold(
    channel_systems: np.ndarray,
    nparticles: int,
    inphase: np.ndarray,
    quadrature: np.ndarray,
    coherent: bool,
):
    systems = np.unique(channel_systems)
    sys_powers = np.ones(nparticles)
    weights = sys_powers / nparticles

    for system in systems:
        sys_indices = (channel_systems == system).nonzero()

        sys_inphase = inphase.T[sys_indices]
        sys_quadrature = quadrature.T[sys_indices]

        if coherent:
            # sys_power = np.sum(sys_inphase + sys_quadrature, axis=0) ** 2
            ipower = np.sum(inphase.T, axis=0) ** 6
            qpower = np.sum(quadrature.T, axis=0) ** 6
            sys_powers = ipower + qpower

        else:
            ipower = np.sum(sys_inphase**2, axis=0)
            qpower = np.sum(sys_quadrature**2, axis=0)

        sys_power = ipower + qpower

        sys_powers *= sys_power

    weights = weights * sys_powers
    # weights = weights - np.max(weights)
    scaled_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    # norm_weights = weights / np.sum(weights)

    return scaled_weights


def main():
    # generate east-north grid
    spread_dimensions = create_spread_grid(
        delta=NE_DELTA, nspheres=NSPHERES, ntiles=NTILES
    )
    nparticles = int((spread_dimensions.size / 2) ** 2)

    enu_grid = np.array(list(itertools.product(*spread_dimensions)))
    enu_grid = np.hstack((enu_grid, np.expand_dims(np.zeros(nparticles), 1)))

    # create simulations
    conf = ns.get_configuration(configuration_path=CONFIG_PATH)

    meas_sim = ns.get_signal_simulation(
        simulation_type="measurement",
        configuration=conf,
    )
    corr_sim = ns.CorrelatorSimulation(configuration=conf)

    rx_pos, rx_vel = eh.prepare_trajectories(
        file_path=TRAJECTORY_PATH.with_suffix(".csv"), fsim=conf.time.fsim
    )  # upsamples trajectories to fsim

    meas_sim.generate_truth(rx_pos=rx_pos, rx_vel=rx_vel)
    meas_sim.simulate()

    emitter_states = meas_sim.emitter_states.truth[0]
    observables = meas_sim.observables[0]
    rx_states = meas_sim.rx_states

    pos = rx_states.pos[0]
    vel = rx_states.vel[0]
    cb = rx_states.clock_bias[0]
    cd = rx_states.clock_drift[0]

    print(f"Position: {rx_states.pos}")
    print(f"Velocity: {rx_states.vel}")
    print(f"Clock Bias: {rx_states.clock_bias}")
    print(f"Clock Drift: {rx_states.clock_drift}")

    ecef_pos0 = rx_states.pos.flatten()
    lla = ecef2lla(x=ecef_pos0[0], y=ecef_pos0[1], z=ecef_pos0[2])

    ecef_pos = np.array(
        enu2ecef(
            east=enu_grid.T[0],
            north=enu_grid.T[1],
            up=enu_grid.T[2],
            lat0=lla.lat,
            lon0=lla.lon,
            h0=lla.alt,
        )
    ).T

    # ecef_pos = np.broadcast_to(pos, ecef_pos.shape)
    ecef_vel = np.broadcast_to(vel, ecef_pos.shape)

    # clock_bias = np.repeat(cb, nparticles)
    clock_bias = enu_grid.T[0] * 100
    clock_drift = np.repeat(cd, nparticles)

    channel_systems, pranges, prange_rates = predict_observables(
        emitter_states=emitter_states,
        rx_pos=ecef_pos,
        rx_vel=ecef_vel,
        rx_clock_bias=clock_bias,
        rx_clock_drift=clock_drift,
    )

    corr_sim.compute_errors(
        observables=observables,
        est_pranges=pranges,
        est_prange_rates=prange_rates,
    )
    corr = corr_sim.correlate(include_subcorrelators=False, include_noise=False)

    manifold_weights = (
        compute_manifold(
            channel_systems=channel_systems,
            nparticles=nparticles,
            inphase=corr.inphase,
            quadrature=corr.quadrature,
            coherent=IS_COHERENT,
        )
        ** 2
    )

    max_idx = np.argmax(manifold_weights)

    plt.figure()
    # plt.plot(clock_bias, np.sum(corr.inphase, axis=1))
    # plt.plot(clock_bias, np.sum(corr.quadrature, axis=1))

    yup = np.sum(corr.inphase + corr.quadrature, axis=1) ** 2
    yup_weights = (yup - np.min(yup)) / (np.max(yup) - np.min(yup))

    plt.plot(
        clock_bias,
        yup_weights,
    )
    plt.plot(
        clock_bias,
        manifold_weights,
    )
    # c = np.sum(corr.inphase + corr.quadrature, axis=1) ** 2
    # cnorm = (c - np.min(c)) / (np.max(c) - np.min(c))

    # nc = np.sum(corr.inphase**2, axis=1) + np.sum(corr.quadrature**2, axis=1)
    # ncnorm = (nc - np.min(nc)) / (np.max(nc) - np.min(nc))

    # plt.plot(corr.inphase.T[0], corr.quadrature.T[0], ".")
    # plt.plot(corr.inphase, ".", c="red", label="In-Phase")
    # plt.plot(corr.quadrature, ".", c="blue", label="Quadrature")
    # plt.plot(
    #     np.sum(corr.inphase, axis=1),
    #     c="purple",
    #     label="Quadrature",
    # )
    # plt.plot(
    #     np.sum(corr.quadrature, axis=1),
    #     c="green",
    #     label="Quadrature",
    # )
    # plt.xlim([int(nparticles / 2) - 1, int(nparticles / 2) + 1])
    # plt.xlim([max_idx - 10, max_idx + 10])
    # plt.xlabel("Candidate States")
    # plt.ylabel("Correlator Output")
    # # plt.legend()

    # plt.figure()
    # plt.plot(
    #     ncnorm,
    #     "-",
    #     c="purple",
    # )
    # plt.plot(
    #     cnorm,
    #     "-",
    #     c="red",
    # )
    # plt.plot(corr.quadrature, c="blue")
    # plt.plot(np.sum(corr.quadrature, axis=1) ** 2, "-", c="orange")
    # plt.plot(np.sum(corr.quadrature**2, axis=1), "-", c="blue")
    # plt.plot(corr.quadrature, c="blue")

    # plt.xlim([max_idx - 10000, max_idx + 10000])

    print(max_idx)
    # print(f"Estimate Position Offset: {enu_grid[max_idx]}")
    surface_plot_weights = manifold_weights.reshape(
        int(np.sqrt(nparticles)), int(np.sqrt(nparticles))
    )

    sns.set_context(CONTEXT)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    east, north = np.meshgrid(spread_dimensions[0], spread_dimensions[1])

    if IS_SMOOTHED:
        east_new, north_new = np.mgrid[
            spread_dimensions[0][0] : spread_dimensions[0][-1] : INTERP_RESOLUTION,
            spread_dimensions[0][0] : spread_dimensions[0][-1] : INTERP_RESOLUTION,
        ]
        tck = bisplrep(east, north, surface_plot_weights, s=0)
        weights_new = bisplev(east_new[:, 0], north_new[0, :], tck)

        surf = ax.plot_surface(
            east_new, north_new, weights_new, cmap=CMAP, linewidth=1, vmin=0, vmax=1
        )

        cset = ax.contour(
            east_new,
            north_new,
            weights_new,
            zdir="z",
            offset=-1,
            cmap=CMAP,
        )
        ax.set_zlim(-1, 1)

    else:
        surf = ax.plot_surface(
            east,
            north,
            surface_plot_weights,
            cmap=CMAP,
            linewidth=1,
            vmin=0,
            vmax=1,
        )

        cset = ax.contour(
            east,
            north,
            surface_plot_weights,
            zdir="z",
            offset=-1,
            cmap=CMAP,
        )
        ax.set_zlim(-1, 1)

    ax.set_facecolor("white")
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1, label="weights")

    ax.view_init(15, 40)
    fig.tight_layout()

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zticklabels([])

    # plt.figure()
    # plt.plot(clock_bias, manifold_weights, ".")
    # plt.xlabel("Clock Bias Error [m]")

    print(enu_grid.T[0].size)

    plt.show()


if __name__ == "__main__":
    main()
