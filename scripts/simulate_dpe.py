import numpy as np
import navsim as ns

from tqdm import tqdm
from pathlib import Path
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators import DirectPositioningConfiguration
from eleventh_hour.navigators.dpe import DirectPositioning

import matplotlib.pyplot as plt

# sim parameters
TRAJECTORY = "daytona_500_sdx_1s_loop"
IS_STATIC = False
IS_EMITTER_TYPE_TRUTH = True

# dpe parameters
POS_SPACINGS = [0.5]
BIAS_SPACINGS = [0.5]
POSTIME_NSTATES = [7]
VEL_SPACINGS = [2, 5]
DRIFT_SPACINGS = [0.25, 0.75]
VELDRIFT_NSTATES = [7, 3]

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

    conf = DirectPositioningConfiguration(
        cn0=cn0,
        rx_pos=rx_pos0,
        rx_vel=rx_vel0,
        rx_clock_bias=rx_clock_bias0,
        rx_clock_drift=rx_clock_drift0,
        rx_clock_type=rx_clock_type,
        T=1 / conf.time.fsim,
        sig_properties=signal_properties,
        pos_spacings=POS_SPACINGS,
        bias_spacings=BIAS_SPACINGS,
        posbias_nstates=POSTIME_NSTATES,
        vel_spacings=VEL_SPACINGS,
        drift_spacings=DRIFT_SPACINGS,
        veldrift_nstates=VELDRIFT_NSTATES,
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
        true_states = np.array(
            [
                sim_rx_states.pos[epoch][0],
                sim_rx_states.pos[epoch][1],
                sim_rx_states.pos[epoch][2],
                sim_rx_states.clock_bias[epoch],
                sim_rx_states.vel[epoch][0],
                sim_rx_states.vel[epoch][1],
                sim_rx_states.vel[epoch][2],
                sim_rx_states.clock_drift[epoch],
            ]
        )

        # limits number of correlations evaluated for speed
        dpe.filter_grids(true_states=true_states, errbuff_pct=1000, is_filtered=True)

        # current estimate observables
        rx_est_pranges, rx_est_prange_rates = dpe.predict_rx_observables(
            emitter_states=emitter_states[epoch]
        )

        # position & bias manifold
        est_pranges, est_prange_rates = dpe.predict_grid_observables(
            emitter_states=emitter_states[epoch]
        )
        corr_sim.compute_errors(
            observables=observables,
            est_pranges=est_pranges,
            est_prange_rates=np.broadcast_to(rx_est_prange_rates, est_pranges.shape),
        )
        pbcorr = corr_sim.correlate(include_subcorrelators=False)
        dpe.estimate_pb(
            inphase=pbcorr.inphase,
            quadrature=pbcorr.quadrature,
        )

        # velocity & drift manifold
        corr_sim.compute_errors(
            observables=observables,
            est_pranges=np.broadcast_to(rx_est_pranges, est_prange_rates.shape),
            est_prange_rates=est_prange_rates,
        )
        vdcorr = corr_sim.correlate(include_subcorrelators=False)
        dpe.estimate_vd(
            inphase=vdcorr.inphase,
            quadrature=vdcorr.quadrature,
        )

        dpe.log_rx_state()

        dpe.time_update()

    states = dpe.rx_states
    perr = states.pos - sim_rx_states.pos.T
    verr = states.vel - sim_rx_states.vel.T

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


if __name__ == "__main__":
    conf, meas_sim, corr_sim = setup_simulation()
    simulate(conf=conf, meas_sim=meas_sim, corr_sim=corr_sim)
