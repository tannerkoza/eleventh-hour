import numpy as np
import navsim as ns
import navtools as nt
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from eleventh_hour.navigators import *
from navtools.conversions import ecef2lla

from eleventh_hour.navigators.vt import VDFLL
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators import VDFLLConfiguration
from eleventh_hour.plot import (
    geoplot,
    skyplot,
    plot_states,
    plot_errors,
    plot_covariances,
    plot_channel_errors,
    plot_correlators,
    plot_cn0s,
)

# sim parameters
TRAJECTORY = "daytona_500_sdx_1s_loop"
IS_STATIC = False
IS_EMITTER_TYPE_TRUTH = True

# vdfll parameters
PROCESS_NOISE_SIGMA = 80
CORRELATOR_BUFF_SIZE = 250
TAP_SPACING = 0.5
NORM_INNOVATION_THRESH = 2.5
NSUBCORRELATORS = 10

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
    pos0 = sim_rx_states.pos[0]
    vel0 = sim_rx_states.vel[0]
    clock_bias0 = sim_rx_states.clock_bias[0]
    clock_drift0 = sim_rx_states.clock_drift[0]
    cn0 = np.array([emitter.cn0 for emitter in sim_observables[0].values()])
    rx_clock_type = conf.errors.rx_clock
    tap_spacings = [0, TAP_SPACING, -TAP_SPACING]

    conf = VDFLLConfiguration(
        proc_noise_sigma=PROCESS_NOISE_SIGMA,
        tap_spacing=TAP_SPACING,
        ni_threshold=NORM_INNOVATION_THRESH,
        cn0_buff_size=CORRELATOR_BUFF_SIZE,
        rx_pos=pos0,
        rx_vel=vel0,
        rx_clock_bias=clock_bias0,
        rx_clock_drift=clock_drift0,
        cn0=cn0,
        rx_clock_type=rx_clock_type,
        sig_properties=signal_properties,
        T=1 / conf.time.fsim,
    )
    vdfll = VDFLL(conf=conf)

    # simulate
    for epoch, observables in tqdm(
        enumerate(sim_observables),
        total=len(sim_observables),
        desc="[eleventh-hour] simulating correlators",
        disable=disable_progress,
    ):
        est_pranges, est_prange_rates = vdfll.predict_observables(
            emitter_states=emitter_states[epoch]
        )

        corr_sim.compute_errors(
            observables=observables,
            est_pranges=est_pranges,
            est_prange_rates=est_prange_rates,
        )
        corr_sim.log_errors()

        correlators = [
            corr_sim.correlate(tap_spacing=tap_spacing, nsubcorrelators=NSUBCORRELATORS)
            for tap_spacing in tap_spacings
        ]

        vdfll.update_correlator_buffers(
            prompt=correlators[0], early=correlators[1], late=correlators[2]
        )
        vdfll.loop_closure()

        # logging
        vdfll.log_rx_state()
        vdfll.log_covariance()
        vdfll.log_cn0()

        vdfll.time_update(T=conf.T)

    # process results
    lla = np.array(
        ecef2lla(
            x=sim_rx_states.pos.T[0],
            y=sim_rx_states.pos.T[1],
            z=sim_rx_states.pos.T[2],
        )
    )
    states, errors = process_state_results(
        truth=sim_rx_states, rx_states=vdfll.rx_states
    )
    covariances = process_covariance_results(
        time=states.time, cov=vdfll.covariances, lat=lla[0], lon=lla[1]
    )

    results = SimulationResults(
        states=states,
        errors=errors,
        covariances=covariances,
        chip_errors=corr_sim.chip_errors,
        ferrors=corr_sim.ferrors,
        prange_errors=corr_sim.code_prange_errors,
        prange_rate_errors=corr_sim.prange_rate_errors,
        cn0s=vdfll.cn0s,
        channel_errors=vdfll.channel_errors,
        correlators=vdfll.correlators,
    )

    return results, sim_emitter_states.truth


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
    results, truth_emitter_states = simulate(
        conf=conf, meas_sim=meas_sim, corr_sim=corr_sim
    )

    now = datetime.now().strftime(format="%Y%m%d-%H%M%S")
    output_dir = (
        DATA_PATH
        / "figures"
        / f"{now}_VP_eleventh-hour_{TRAJECTORY}_{conf.time.fsim}Hz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: clean this nonsense

    plt.close()
    plt.figure()
    azimuth = defaultdict(lambda: defaultdict(lambda: []))
    elevation = defaultdict(lambda: defaultdict(lambda: []))

    for epoch in truth_emitter_states:
        for emitter in epoch.values():
            azimuth[emitter.constellation][emitter.id].append(emitter.az)
            elevation[emitter.constellation][emitter.id].append(emitter.el)

    for constellation in azimuth.keys():
        az = np.array(nt.pad_list(list(azimuth[constellation].values())))
        el = np.array(nt.pad_list(list(elevation[constellation].values())))
        names = list(azimuth[constellation].keys())

        skyplot(
            az=az,
            el=el,
            label=constellation,
            name=names,
            deg=False,
        )

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "skyplot")

    geoplot(
        lat=results.states.truth_lla[0],
        lon=results.states.truth_lla[1],
        output_dir=output_dir,
    )
    plot_states(states=results.states, output_dir=output_dir)
    plot_errors(errors=results.errors, output_dir=output_dir)
    plot_covariances(cov=results.covariances, output_dir=output_dir)
    plot_channel_errors(
        time=results.states.time,
        channel_errors=results.channel_errors,
        output_dir=output_dir,
    )
    plot_correlators(
        time=results.states.time, correlators=results.correlators, output_dir=output_dir
    )
    plot_cn0s(time=results.states.time, cn0s=results.cn0s, output_dir=output_dir)
