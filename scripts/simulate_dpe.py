import numpy as np
import navsim as ns

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from eleventh_hour.navigators import *
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators.dpe import DirectPositioning

import matplotlib.pyplot as plt
from eleventh_hour.plot import (
    geoplot,
    pf_animation,
    plot_states,
    plot_errors,
    plot_covariances,
)

# sim parameters
TRAJECTORY = "flight_australia_sdx_1s_onego"
IS_STATIC = False
IS_EMITTER_TYPE_TRUTH = True

# dpe parameters
PROCESS_NOISE_SIGMA = 6
DELAY_BIAS_SIGMA = 15
DRIFT_BIAS_SIGMA = 3
DELAY_BIAS_RESOLUTION = 0.015
DRIFT_BIAS_RESOLUTION = 0.015

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

    if IS_EMITTER_TYPE_TRUTH:
        emitter_states = sim_emitter_states.truth
    else:
        emitter_states = sim_emitter_states.ephemeris

    # navigator setup
    rx_pos0 = sim_rx_states.pos[0]
    rx_vel0 = sim_rx_states.vel[0]
    rx_clock_bias0 = sim_rx_states.clock_bias[0]
    rx_clock_drift0 = sim_rx_states.clock_drift[0]
    rx_clock_type = conf.errors.rx_clock

    conf = DPEConfiguration(
        is_grid=True,
        delay_bias_sigma=DELAY_BIAS_SIGMA,
        delay_bias_resolution=DELAY_BIAS_RESOLUTION,
        drift_bias_resolution=DRIFT_BIAS_RESOLUTION,
        drift_bias_sigma=DRIFT_BIAS_SIGMA,
        process_noise_sigma=PROCESS_NOISE_SIGMA,
        neff_percentage=50.0,
        rx_pos=rx_pos0,
        rx_vel=rx_vel0,
        rx_clock_bias=rx_clock_bias0,
        rx_clock_drift=rx_clock_drift0,
        rx_clock_type=rx_clock_type,
        T=1 / conf.time.fsim,
    )

    dpe = DirectPositioning(conf=conf)

    # simulate
    for epoch, observables in tqdm(
        enumerate(sim_observables),
        total=len(sim_observables),
        desc="[eleventh-hour] simulating correlators",
        disable=disable_progress,
    ):
        particle_pranges, particle_prange_rates = dpe.predict_particle_observables(
            emitter_states=emitter_states[epoch]
        )

        corr_sim.compute_errors(
            observables=observables,
            est_pranges=particle_pranges,
            est_prange_rates=particle_prange_rates,
        )
        corr = corr_sim.correlate(include_subcorrelators=False)

        dpe.estimate_state(inphase=corr.inphase, quadrature=corr.quadrature)

        dpe.time_update(T=conf.T)

        # logging
        est_pranges, est_prange_rates = dpe.predict_estimate_observables(
            emitter_states=emitter_states[epoch]
        )
        corr_sim.compute_errors(
            observables=observables,
            est_pranges=est_pranges,
            est_prange_rates=est_prange_rates,
        )  # computes current estimates errors
        corr_sim.log_errors()

        dpe.log_rx_state()
        dpe.log_covariance()

    # process results
    lla = np.array(
        ecef2lla(
            x=sim_rx_states.pos.T[0],
            y=sim_rx_states.pos.T[1],
            z=sim_rx_states.pos.T[2],
        )
    )
    states, errors = process_state_results(
        truth=sim_rx_states, rx_states=dpe.rx_states, particles=dpe.particles
    )
    covariances = process_covariance_results(
        time=states.time, cov=dpe.covariances, lat=lla[0], lon=lla[1]
    )

    results = SimulationResults(
        states=states,
        errors=errors,
        covariances=covariances,
        chip_errors=corr_sim.chip_errors,
        ferrors=corr_sim.ferrors,
        prange_errors=corr_sim.code_prange_errors,
        prange_rate_errors=corr_sim.prange_rate_errors,
    )

    return results


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
    results = simulate(conf=conf, meas_sim=meas_sim, corr_sim=corr_sim)

    now = datetime.now().strftime(format="%Y%m%d-%H%M%S")
    output_dir = (
        DATA_PATH
        / "figures"
        / f"{now}_DPE_eleventh-hour_{TRAJECTORY}_{conf.time.fsim}Hz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pf_animation(
        time=results.states.time,
        particles=results.states.particles.pos,
        weights=results.states.particles.weights,
        truth=results.states.truth_enu_pos,
        rx=results.states.enu_pos,
        output_dir=output_dir,
    )
    plt.figure()
    # geoplot(
    #     lat=results.states.truth_lla[0],
    #     lon=results.states.truth_lla[1],
    #     output_dir=output_dir,
    # )
    plot_states(states=results.states, output_dir=output_dir, label="dpe")
    plot_errors(errors=results.errors, output_dir=output_dir, label="dpe")
    plot_covariances(cov=results.covariances, output_dir=output_dir, label="dpe")
