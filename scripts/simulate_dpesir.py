import numpy as np
import navsim as ns
import navtools as nt

from tqdm import tqdm
from pathlib import Path
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators import DPEConfiguration
from eleventh_hour.navigators.dpe import DirectPositioning

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# sim parameters
TRAJECTORY = "daytona_500_sdx_1s_loop"
IS_STATIC = False
IS_EMITTER_TYPE_TRUTH = True

# dpe parameters
NSPHERES = 200
PDELTA = 7.5
VDELTA = 5.5
BDELTA = 1.5
DDELTA = 0.25
PROCESS_NOISE_SIGMA = 10
# plot parameters
SKYPLOT_PERIOD = 5  # [s]

# path
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
TRAJECTORY_PATH = DATA_PATH / "trajectories" / TRAJECTORY


def setup_simulation(disable_progress: bool = False):
    # sim setup
    conf = ns.get_configuration(configuration_path=CONFIG_PATH)

    rx_pos, rx_vel = prepare_trajectories(
        file_path=TRAJECTORY_PATH.with_suffix(".csv"), fsim=conf.time.fsim
    )  # upsamples trajectories to fsim

    if IS_STATIC:
        rx_pos = rx_pos[0]
        rx_vel = np.zeros_like(rx_pos)

    meas_sim = __generate_truth(
        conf=conf, rx_pos=rx_pos, rx_vel=rx_vel, disable_progress=disable_progress
    )
    corr_sim = ns.CorrelatorSimulation(configuration=conf)

    return conf, meas_sim, corr_sim


def simulate(
    conf: ns.SimulationConfiguration,
    meas_sim: ns.MeasurementSimulation,
    corr_sim: ns.CorrelatorSimulation,
    disable_progress: bool = False,
):
    # measurement simulation
    meas_sim.simulate()

    sim_emitter_states = meas_sim.emitter_states
    sim_observables = meas_sim.observables
    sim_rx_states = meas_sim.rx_states
    signal_properties = meas_sim.signal_properties

    if IS_EMITTER_TYPE_TRUTH:
        emitter_states = sim_emitter_states.truth
    else:
        emitter_states = sim_emitter_states.ephemeris

    # navigator setup
    rx_pos0 = sim_rx_states.pos[0]
    rx_vel0 = sim_rx_states.vel[0]
    rx_clock_bias0 = sim_rx_states.clock_bias[0]
    rx_clock_drift0 = sim_rx_states.clock_drift[0]
    cn0 = np.array([emitter.cn0 for emitter in sim_observables[0].values()])
    rx_clock_type = conf.errors.rx_clock

    conf = DPEConfiguration(
        nspheres=NSPHERES,
        pdelta=PDELTA,
        vdelta=VDELTA,
        bdelta=BDELTA,
        ddelta=DDELTA,
        process_noise_sigma=PROCESS_NOISE_SIGMA,
        cn0=cn0,
        rx_pos=rx_pos0,
        rx_vel=rx_vel0,
        rx_clock_bias=rx_clock_bias0,
        rx_clock_drift=rx_clock_drift0,
        rx_clock_type=rx_clock_type,
        T=1 / conf.time.fsim,
        sig_properties=signal_properties,
    )

    dpe = DirectPositioning(conf=conf)

    # simulate
    rx_states = []
    for epoch, observables in tqdm(
        enumerate(sim_observables),
        total=len(sim_observables),
        desc="[eleventh-hour] simulating correlators",
        disable=disable_progress,
    ):
        dpe.log_particles()
        est_pranges, est_prange_rates = dpe.__predict_observables(
            emitter_states=emitter_states[epoch]
        )

        corr_sim.compute_errors(
            observables=observables,
            est_pranges=est_pranges,
            est_prange_rates=est_prange_rates,
        )
        corr = corr_sim.correlate(include_subcorrelators=False)

        dpe.estimate_state(inphase=corr.inphase, quadrature=corr.quadrature)
        dpe.log_rx_state()
        dpe.time_update(T=conf.T)

    states = dpe.rx_states
    particles = dpe.epoch_particles
    perr = states.pos - sim_rx_states.pos.T
    verr = states.vel - sim_rx_states.vel.T

    pf_animation(
        particles=particles,
        truth=sim_rx_states,
        rx=states,
        T=conf.T,
        frames=len(sim_observables),
    )

    plt.figure()
    plt.plot(perr.T)
    plt.figure()
    plt.plot(verr.T)
    plt.show()


# private
def __generate_truth(
    conf: ns.SimulationConfiguration,
    rx_pos: np.ndarray,
    rx_vel: np.ndarray = None,
    disable_progress: bool = False,
):
    meas_sim = ns.get_signal_simulation(
        simulation_type="measurement",
        configuration=conf,
        disable_progress=disable_progress,
    )
    meas_sim.generate_truth(rx_pos=rx_pos, rx_vel=rx_vel)

    return meas_sim


def pf_animation(particles, truth, rx, T, frames):
    fig, ax = plt.subplots()

    lla0 = nt.ecef2lla(truth.pos[0, 0], truth.pos[0, 1], truth.pos[0, 2])
    particles = np.array(
        nt.ecef2enu(
            x=particles.pos[:, 0],
            y=particles.pos[:, 1],
            z=particles.pos[:, 2],
            lat0=lla0.lat,
            lon0=lla0.lon,
            alt0=lla0.alt,
        )
    )
    truth = np.array(
        nt.ecef2enu(
            x=truth.pos[:, 0],
            y=truth.pos[:, 1],
            z=truth.pos[:, 2],
            lat0=lla0.lat,
            lon0=lla0.lon,
            alt0=lla0.alt,
        )
    ).T
    rx = np.array(
        nt.ecef2enu(
            x=rx.pos[:, 0],
            y=rx.pos[:, 1],
            z=rx.pos[:, 2],
            lat0=lla0.lat,
            lon0=lla0.lon,
            alt0=lla0.alt,
        )
    ).T

    part = ax.scatter(particles[0, 0], particles[1, 0], c="b", s=5, label="particles")
    # t = ax.plot(truth[0, 0], truth[0, 1], "g*", label="truth")[0]
    # r = ax.plot(rx[0, 0], rx[0, 1], "*", label="rx")[0]
    ax.legend()

    def update(frame):
        # for each frame, update the data stored on each artist.
        x = particles[0, frame]
        y = particles[1, frame]
        # # update the scatter plot:
        data = np.stack([x, y]).T
        part.set_offsets(data)
        # update the line plot:
        # t.set_xdata(truth[frame, 0])
        # t.set_ydata(truth[frame, 1])
        # r.set_xdata(rx[frame, 0])
        # r.set_ydata(rx[frame, 1])

        return part

    ani = animation.FuncAnimation(fig=fig, func=update, interval=T, frames=frames)
    plt.show()


if __name__ == "__main__":
    conf, meas_sim, corr_sim = setup_simulation()
    simulate(conf=conf, meas_sim=meas_sim, corr_sim=corr_sim)
