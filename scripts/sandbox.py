import numpy as np
import navsim as ns
import navtools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import eleventh_hour.trajectories as eh

from pathlib import Path
from collections import defaultdict
from navtools.conversions import ecef2lla
from eleventh_hour.plot import (
    geoplot,
    skyplot,
)

TRAJECTORY = "flight_australia_sdx_1s_onego"

# dpe
IS_COHERENT = False
NE_DELTA = 1
NSPHERES = 500
NTILES = 2

# plotting
CONTEXT = "paper"
COLORS = ["#DB0000", "#00A9C1"]
sns.set_palette(sns.color_palette(COLORS))


# path literals
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
TRAJECTORY_PATH = DATA_PATH / "trajectories" / TRAJECTORY


def create_skyplot(emitter_states: list):
    azimuth = defaultdict(lambda: defaultdict(lambda: []))
    elevation = defaultdict(lambda: defaultdict(lambda: []))

    for epoch in emitter_states:
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
            label=constellation.lower(),
            name=names,
            deg=False,
            s=10,
        )


def main():
    # create simulation
    conf = ns.get_configuration(configuration_path=CONFIG_PATH)

    meas_sim = ns.get_signal_simulation(
        simulation_type="measurement",
        configuration=conf,
    )

    rx_pos, rx_vel = eh.prepare_trajectories(
        file_path=TRAJECTORY_PATH.with_suffix(".csv"), fsim=conf.time.fsim
    )  # upsamples trajectories to fsim

    meas_sim.generate_truth(rx_pos=rx_pos, rx_vel=rx_vel)
    meas_sim.simulate()

    emitter_states = meas_sim.emitter_states.truth
    dop = meas_sim.emitter_states.dop
    observables = meas_sim.observables
    rx_states = meas_sim.rx_states

    gdop = np.array([np.linalg.norm(np.diag(epoch)) for epoch in dop])
    tdop = np.array([np.diag(epoch)[3] for epoch in dop])
    hdop = np.array([np.linalg.norm(np.diag(epoch)[:2]) for epoch in dop])
    vdop = np.array([np.diag(epoch)[2] for epoch in dop])

    create_skyplot(emitter_states=emitter_states)
    plt.show()

    speed = np.max(np.linalg.norm(rx_states.vel, axis=1))
    print(f"Max Speed: {speed}")

    # lla = ecef2lla(x=rx_states.pos.T[0], y=rx_states.pos.T[1], z=rx_states.pos.T[2])
    # plt.figure()
    # geoplot(
    #     lat=lla.lat,
    #     lon=lla.lon,
    # )


if __name__ == "__main__":
    main()
