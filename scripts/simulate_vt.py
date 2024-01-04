import numpy as np
import navsim as ns
import pymap3d as pm
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators.vt import VDFLL, VDFLLConfiguration

# user-defined parameters
IS_TRAJECTORY = True
IS_EMITTER_TYPE_TRUTH = False
IS_TQDM_DISABLED = False

PROCESS_NOISE_SIGMA = 20
CORRELATOR_BUFF_SIZE = 20
TAP_SPACING = 0.5
NSUBCORRELATORS = [4, 0, 0]
TRAJECTORY = "daytona_500_sdx_1s_loop"
RX_POS = np.array([423756, -5361363, 3417705])

# path
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
TRAJECTORY_PATH = DATA_PATH / "trajectories" / TRAJECTORY


def simulate():
    # sim setup
    conf = ns.get_configuration(configuration_path=CONFIG_PATH)

    if IS_TRAJECTORY:
        rx_pos, rx_vel = prepare_trajectories(
            file_path=TRAJECTORY_PATH.with_suffix(".csv"), fsim=conf.time.fsim
        )

        ecef0 = rx_pos[0]
        lla0 = pm.ecef2geodetic(x=ecef0[0], y=ecef0[1], z=ecef0[2])
    else:
        rx_pos = RX_POS
        rx_vel = np.zeros_like(RX_POS)
        lla0 = pm.ecef2geodetic(x=rx_pos[0], y=rx_pos[1], z=rx_pos[2])

    sim = ns.CorrelatorSimulation(configuration=conf)
    (
        sim_emitter_states,
        sim_observables,
        sim_rx_states,
        signal_properties,
    ) = simulate_measurements(conf=conf, rx_pos=rx_pos, rx_vel=rx_vel)

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
    tap_spacings = [0, TAP_SPACING, -TAP_SPACING]

    vdfll_conf = VDFLLConfiguration(
        process_noise_sigma=PROCESS_NOISE_SIGMA,
        tap_spacing=TAP_SPACING,
        correlator_buff_size=CORRELATOR_BUFF_SIZE,
        rx_pos=rx_pos0,
        rx_vel=rx_vel0,
        rx_clock_bias=rx_clock_bias0,
        rx_clock_drift=rx_clock_drift0,
        cn0=cn0,
        rx_clock_type=rx_clock_type,
        signal_properties=signal_properties,
    )
    vdfll = VDFLL(conf=vdfll_conf)

    # simulate
    for epoch, observables in tqdm(
        enumerate(sim_observables),
        total=len(sim_observables),
        desc="[eleventh-hour] simulating correlators",
        disable=IS_TQDM_DISABLED,
    ):
        vdfll.time_update(T=1 / conf.time.fsim)
        est_pranges, est_prange_rates = vdfll.predict_observables(
            emitter_states=emitter_states[epoch]
        )

        sim.compute_errors(
            observables=observables,
            est_pranges=est_pranges,
            est_prange_rates=est_prange_rates,
        )
        correlators = [
            sim.correlate(tap_spacing=tap_spacing, nsubcorrelators=nsubcorrelators)
            for tap_spacing, nsubcorrelators in zip(tap_spacings, NSUBCORRELATORS)
        ]

        vdfll.update_correlator_buffers(
            prompt=correlators[0], early=correlators[1], late=correlators[2]
        )
        vdfll.cn0 = np.array([emitter.cn0 for emitter in observables.values()])
        vdfll.loop_closure()

    # plot
    plot(truth_states=sim_rx_states, vdfll=vdfll, lla0=lla0)


def simulate_measurements(
    conf: ns.SimulationConfiguration,
    rx_pos: np.ndarray,
    rx_vel: np.ndarray = None,
):
    meas_sim = ns.get_signal_simulation(
        simulation_type="measurement", configuration=conf
    )
    meas_sim.simulate(rx_pos=rx_pos, rx_vel=rx_vel)

    return (
        meas_sim.emitter_states,
        meas_sim.observables,
        meas_sim.rx_states,
        meas_sim.signal_properties,
    )


def plot(truth_states: ns.ReceiverTruthStates, vdfll: VDFLL, lla0: np.ndarray):
    truth_pos = np.array(
        pm.ecef2enu(
            x=truth_states.pos[:, 0],
            y=truth_states.pos[:, 1],
            z=truth_states.pos[:, 2],
            lat0=lla0[0],
            lon0=lla0[1],
            h0=lla0[2],
        )
    )
    truth_vel = np.array(
        pm.ecef2enuv(
            u=truth_states.vel[:, 0],
            v=truth_states.vel[:, 1],
            w=truth_states.vel[:, 2],
            lat0=lla0[0],
            lon0=lla0[1],
        )
    )

    vdfll_pos = np.array(
        pm.ecef2enu(
            x=vdfll.rx_states[:, 0],
            y=vdfll.rx_states[:, 2],
            z=vdfll.rx_states[:, 4],
            lat0=lla0[0],
            lon0=lla0[1],
            h0=lla0[2],
        )
    )
    vdfll_vel = np.array(
        pm.ecef2enuv(
            u=vdfll.rx_states[:, 1],
            v=vdfll.rx_states[:, 3],
            w=vdfll.rx_states[:, 5],
            lat0=lla0[0],
            lon0=lla0[1],
        )
    )
    vdfll_cb = vdfll.rx_states[:, 6]
    vdfll_cd = vdfll.rx_states[:, 7]

    pos_error = truth_pos.T - vdfll_pos.T
    vel_error = truth_vel.T - vdfll_vel.T
    cb_error = truth_states.clock_bias - vdfll_cb
    cd_error = truth_states.clock_drift - vdfll_cd

    plt.figure()
    plt.title("Trajectory")
    plt.plot(truth_pos[0], truth_pos[1], label="truth", marker="*")
    plt.plot(vdfll_pos[0], vdfll_pos[1], label="vdfll", marker="*")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()

    plt.figure()
    plt.title("Position Error [m]")
    plt.plot(truth_states.time, pos_error, label=["x", "y", "z"])
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.legend()

    plt.figure()
    plt.title("Velocity Error [m/s]")
    plt.plot(truth_states.time, vel_error, label=["x", "y", "z"])
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m/s]")
    plt.legend()

    plt.figure()
    plt.title("Clock Bias Error [m]")
    plt.plot(truth_states.time, cb_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")

    plt.figure()
    plt.title("Clock Drift Error [m/s]")
    plt.plot(truth_states.time, cd_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m/s]")

    plt.figure()
    plt.title("Code Discriminator [m]")
    plt.plot(truth_states.time, vdfll.prange_errors)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")

    plt.figure()
    plt.title("Frequency Discriminator [m/s]")
    plt.plot(truth_states.time, vdfll.prange_rate_errors)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m/s]")

    plt.show()


if __name__ == "__main__":
    simulate()
