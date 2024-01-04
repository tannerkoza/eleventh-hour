import numpy as np
import navsim as ns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from eleventh_hour.navigators.vt import VDFLL, VDFLLConfiguration

# user-defined parameters
RX_POS = np.array([423756, -5361363, 3417705])
IS_EMITTER_TYPE_TRUTH = True
TAP_SPACING = [0, 0.5, -0.5]
NSUBCORRELATORS = [4, None, None]
DISABLE_TQDM = False

# path
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"


def simulate():
    # sim setup
    conf = ns.get_configuration(configuration_path=CONFIG_PATH)
    sim = ns.CorrelatorSimulation(configuration=conf)
    (
        sim_emitter_states,
        sim_observables,
        sim_rx_states,
        signal_properties,
    ) = simulate_measurements(conf=conf, rx_pos=RX_POS)

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

    vdfll_conf = VDFLLConfiguration(
        process_noise_sigma=5,
        tap_spacing=0.5,
        correlator_buff_size=20,
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
        disable=DISABLE_TQDM,
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
            for tap_spacing, nsubcorrelators in zip(TAP_SPACING, NSUBCORRELATORS)
        ]

        vdfll.update_correlator_buffers(
            prompt=correlators[0], early=correlators[1], late=correlators[2]
        )
        vdfll.loop_closure()

    # plot
    plot(truth_states=sim_rx_states, vdfll=vdfll)


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


def plot(truth_states: ns.ReceiverTruthStates, vdfll: VDFLL):
    vdfll_pos = vdfll.rx_states[:, :6:2]
    vdfll_vel = vdfll.rx_states[:, 1:7:2]
    vdfll_cb = vdfll.rx_states[:, 6]
    vdfll_cd = vdfll.rx_states[:, 7]

    pos_error = truth_states.pos - vdfll_pos
    vel_error = truth_states.vel - vdfll_vel
    cb_error = truth_states.clock_bias - vdfll_cb
    cd_error = truth_states.clock_drift - vdfll_cd

    plt.figure()
    plt.title("Position Error [m]")
    plt.plot(truth_states.time, pos_error)

    plt.figure()
    plt.title("Velocity Error [m/s]")
    plt.plot(truth_states.time, vel_error)

    plt.figure()
    plt.title("Clock Bias Error [m]")
    plt.plot(truth_states.time, cb_error)

    plt.figure()
    plt.title("Clock Drift Error [m/s]")
    plt.plot(truth_states.time, cd_error)

    plt.show()


if __name__ == "__main__":
    simulate()
