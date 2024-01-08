import numpy as np
import navsim as ns
import pymap3d as pm
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from eleventh_hour.plot import geoplot, skyplot, plot_constellation_dataframe
from eleventh_hour.trajectories import prepare_trajectories
from eleventh_hour.navigators.vt import VDFLL, VDFLLConfiguration

# sim parameters
TRAJECTORY = "road_usa_sdx_1s_onego"
IS_STATIC = False
IS_EMITTER_TYPE_TRUTH = False
IS_PLOTTED = False

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
    )

    if IS_STATIC:
        rx_pos = rx_pos[0]
        rx_vel = np.zeros_like(rx_pos)

    meas_sim = generate_truth(
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

    # plot
    if IS_PLOTTED:
        now = datetime.now().strftime(format="%Y%m%d-%H%M%S")
        output_dir = (
            DATA_PATH
            / "figures"
            / f"{now}_VT_SingleRun_{TRAJECTORY}_{int(conf.time.duration)}s_{conf.time.fsim}Hz"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        plot(
            truth_rx_states=sim_rx_states,
            truth_emitter_states=sim_emitter_states.truth[
                :: SKYPLOT_PERIOD * conf.time.fsim
            ],
            vdfll=vdfll,
            dop=sim_emitter_states.dop,
            output_dir=output_dir,
        )

    return sim_rx_states, vdfll


def generate_truth(
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


def plot(
    truth_rx_states: ns.ReceiverTruthStates,
    truth_emitter_states: list,
    vdfll: VDFLL,
    dop: np.ndarray,
    output_dir: Path,
):
    sns.set_context("paper")

    # unpack data
    azimuth = defaultdict(lambda: defaultdict(lambda: []))
    elevation = defaultdict(lambda: defaultdict(lambda: []))

    for epoch in truth_emitter_states:
        for emitter in epoch.values():
            azimuth[emitter.constellation][emitter.id].append(emitter.az)
            elevation[emitter.constellation][emitter.id].append(emitter.el)

    truth_lla = np.array(
        pm.ecef2geodetic(
            x=truth_rx_states.pos[:, 0],
            y=truth_rx_states.pos[:, 1],
            z=truth_rx_states.pos[:, 2],
        )
    )
    lla0 = truth_lla[:, 0]

    truth_pos = np.array(
        pm.ecef2enu(
            x=truth_rx_states.pos[:, 0],
            y=truth_rx_states.pos[:, 1],
            z=truth_rx_states.pos[:, 2],
            lat0=lla0[0],
            lon0=lla0[1],
            h0=lla0[2],
        )
    )
    truth_vel = np.array(
        pm.ecef2enuv(
            u=truth_rx_states.vel[:, 0],
            v=truth_rx_states.vel[:, 1],
            w=truth_rx_states.vel[:, 2],
            lat0=lla0[0],
            lon0=lla0[1],
        )
    )

    vdfll_pos = np.array(
        pm.ecef2enu(
            x=vdfll.rx_states.x_pos,
            y=vdfll.rx_states.y_pos,
            z=vdfll.rx_states.z_pos,
            lat0=lla0[0],
            lon0=lla0[1],
            h0=lla0[2],
        )
    )
    vdfll_vel = np.array(
        pm.ecef2enuv(
            u=vdfll.rx_states.x_vel,
            v=vdfll.rx_states.y_vel,
            w=vdfll.rx_states.z_vel,
            lat0=lla0[0],
            lon0=lla0[1],
        )
    )
    vdfll_cb = vdfll.rx_states.clock_bias
    vdfll_cd = vdfll.rx_states.clock_drfit

    pos_error = truth_pos.T - vdfll_pos.T
    vel_error = truth_vel.T - vdfll_vel.T
    cb_error = truth_rx_states.clock_bias - vdfll_cb
    cd_error = truth_rx_states.clock_drift - vdfll_cd

    ip = vdfll.pad_log(log=[epoch.ip for epoch in vdfll.correlators])
    qp = vdfll.pad_log(log=[epoch.qp for epoch in vdfll.correlators])

    gdop = []
    pdop = []
    hdop = []
    vdop = []
    for epoch in dop:
        diag = np.diag(epoch)
        gdop.append(np.linalg.norm(diag))
        pdop.append(np.linalg.norm(diag[:3]))
        hdop.append(np.linalg.norm(diag[:2]))
        vdop.append(diag[2])

    gdop = np.array(gdop)
    pdop = np.array(pdop)
    hdop = np.array(hdop)
    vdop = np.array(vdop)

    # plot
    geoplot(lat=truth_lla[0], lon=truth_lla[1])
    plt.tight_layout()
    plt.savefig(fname=output_dir / "geoplot")

    for constellation in azimuth.keys():
        az = np.array(vdfll.pad_log(list(azimuth[constellation].values())))
        el = np.array(vdfll.pad_log(list(elevation[constellation].values())))
        names = list(azimuth[constellation].keys())

        skyplot(
            az=az,
            el=el,
            name=names,
            label=constellation,
            deg=False,
        )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "skyplot")

    plt.figure()
    plt.title("DOP")
    plt.plot(truth_rx_states.time, gdop, label="gdop")
    plt.plot(truth_rx_states.time, pdop, label="pdop")
    plt.plot(truth_rx_states.time, hdop, label="hdop")
    plt.plot(truth_rx_states.time, vdop, label="vdop")
    plt.xlabel("Time [s]")
    plt.ylabel("DOP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname=output_dir / "dop")

    plt.figure()
    plt.title("Trajectory")
    plt.plot(truth_pos[0], truth_pos[1], label="truth", marker="*")
    plt.plot(vdfll_pos[0], vdfll_pos[1], label="vdfll", marker="*")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()
    ax = plt.gca()
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "trajectory")

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.supxlabel("Time [s]")
    fig.supylabel("Position [m]")
    fig.suptitle("Position: East, North, Up")
    for index, ax in enumerate(axes):
        ax.plot(truth_rx_states.time, truth_pos[index], label="truth")
        ax.plot(truth_rx_states.time, vdfll_pos[index], label="vdfll")
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(fname=output_dir / "position")

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.supxlabel("Time [s]")
    fig.supylabel("Velocity [m/s]")
    fig.suptitle("Velocity: East, North, Up")
    for index, ax in enumerate(axes):
        ax.plot(truth_rx_states.time, truth_vel[index], label="truth")
        ax.plot(truth_rx_states.time, vdfll_vel[index], label="vdfll")
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(fname=output_dir / "velocity")

    plt.figure()
    plt.title("Position Error [m]")
    plt.plot(truth_rx_states.time, pos_error, label=["east", "north", "up"])
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname=output_dir / "position_error")

    plt.figure()
    plt.title("Velocity Error [m/s]")
    plt.plot(truth_rx_states.time, vel_error, label=["east", "north", "up"])
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname=output_dir / "velocity_error")

    plt.figure()
    plt.title("Clock Bias Error [m]")
    plt.plot(truth_rx_states.time, cb_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "cb_error")

    plt.figure()
    plt.title("Clock Drift Error [m/s]")
    plt.plot(truth_rx_states.time, cd_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m/s]")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "cd_error")

    plt.figure()
    plt.title("Code Discriminator [m]")
    plot_constellation_dataframe(truth_rx_states.time, vdfll.prange_errors)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "code_discriminator")

    plt.figure()
    plt.title("Frequency Discriminator [m/s]")
    plot_constellation_dataframe(truth_rx_states.time, vdfll.prange_rate_errors)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m/s]")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "f_discriminator")

    plt.figure()
    plt.title("Prompt Correlator Phasor")
    plt.plot(ip, qp, ".")
    plt.xlabel("Inphase Power")
    plt.ylabel("Quadrature Power")
    ax = plt.gca()
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(fname=output_dir / "prompt_corr_phasor")


if __name__ == "__main__":
    conf, meas_sim, corr_sim = setup_simulation()
    simulate(conf=conf, meas_sim=meas_sim, corr_sim=corr_sim)
