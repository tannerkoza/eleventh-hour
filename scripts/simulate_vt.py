import numpy as np
import navsim as ns

from tqdm import tqdm
from pathlib import Path
from eleventh_hour.data import process_vdfll_states
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators.vt import VDFLL, VDFLLConfiguration

# sim parameters
TRAJECTORY = "daytona_500_sdx_1s_loop"
IS_STATIC = False
IS_EMITTER_TYPE_TRUTH = False

# vdfll parameters
PROCESS_NOISE_SIGMA = 50
CORRELATOR_BUFF_SIZE = 20
TAP_SPACING = 0.5
NORM_INNOVATION_THRESH = 2.5
NSUBCORRELATORS = 4

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
    tap_spacings = [0, TAP_SPACING, -TAP_SPACING]

    vdfll_conf = VDFLLConfiguration(
        process_noise_sigma=PROCESS_NOISE_SIGMA,
        tap_spacing=TAP_SPACING,
        norm_innovation_thresh=NORM_INNOVATION_THRESH,
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
        disable=disable_progress,
    ):
        vdfll.time_update(T=1 / conf.time.fsim)
        est_pranges, est_prange_rates = vdfll.predict_observables(
            emitter_states=emitter_states[epoch]
        )

        corr_sim.compute_errors(
            observables=observables,
            est_pranges=est_pranges,
            est_prange_rates=est_prange_rates,
        )
        correlators = [
            corr_sim.correlate(tap_spacing=tap_spacing, nsubcorrelators=NSUBCORRELATORS)
            for tap_spacing in tap_spacings
        ]

        vdfll.update_correlator_buffers(
            prompt=correlators[0], early=correlators[1], late=correlators[2]
        )
        vdfll.cn0 = np.array([emitter.cn0 for emitter in observables.values()])
        vdfll.loop_closure()

    # process results
    states, errors = process_vdfll_states(
        truth=sim_rx_states, rx_states=vdfll.rx_states
    )
    channel_errors = vdfll.channel_errors
    correlators = vdfll.correlators

    return states, errors, channel_errors, correlators


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


if __name__ == "__main__":
    conf, meas_sim, corr_sim = setup_simulation()
    simulate(conf=conf, meas_sim=meas_sim, corr_sim=corr_sim)

    # plot
    #     now = datetime.now().strftime(format="%Y%m%d-%H%M%S")
    #     output_dir = (
    #         DATA_PATH
    #         / "figures"
    #         / f"{now}_VT_SingleRun_{TRAJECTORY}_{int(conf.time.duration)}s_{conf.time.fsim}Hz"
    #     )
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     plot(
    #         truth_rx_states=sim_rx_states,
    #         truth_emitter_states=sim_emitter_states.truth[
    #             :: SKYPLOT_PERIOD * conf.time.fsim
    #         ],
    #         vdfll=vdfll,
    #         dop=sim_emitter_states.dop,
    #         output_dir=output_dir,
    #     )
