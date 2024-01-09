import numpy as np
import navsim as ns
import simulate_vt as vt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from eleventh_hour.data import pickle_objects

# sim parameters
NSIMS = 100
JS = np.arange(0, 5, 5, dtype=float)
INTERFERED_CONSTELLATIONS = ["gps"]
DISABLE_PROGRESS = True

# path
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
MC_PATH = DATA_PATH / "monte_carlos"


def monte_carlo():
    conf, meas_sim, corr_sim = vt.setup_simulation(disable_progress=DISABLE_PROGRESS)
    mc_dir = create_sim_dir(conf=conf)

    for js in JS:
        # set current js
        for constellation, properties in conf.constellations.emitters.items():
            if constellation in INTERFERED_CONSTELLATIONS:
                properties.js = js

        js_path = mc_dir / f"js{int(js)}"

        # perform monte carlos
        for sim in tqdm(range(NSIMS), desc=f"simulating with {js} dB J/S"):
            states, errors, channel_errors, correlators = vt.simulate(
                conf=conf,
                meas_sim=meas_sim,
                corr_sim=corr_sim,
                disable_progress=DISABLE_PROGRESS,
            )
            data = {
                "states": states,
                "errors": errors,
                "channel_errors": channel_errors,
                "correlators": correlators,
            }
            output_path = js_path / f"sim{sim}"
            output_path.mkdir(parents=True, exist_ok=True)

            pickle_objects(data=data, output_path=output_path)

            meas_sim.clear_observables()


def create_sim_dir(conf: ns.SignalConfiguration):
    now = datetime.now().strftime(format="%Y%m%d-%H%M%S")

    dir_name = f"{now}_VT_MonteCarlo_{vt.TRAJECTORY}_{int(conf.time.duration)}s_{int(conf.time.fsim)}Hz"
    sim_dir = MC_PATH / dir_name

    sim_dir.mkdir(parents=True, exist_ok=True)

    return sim_dir


if __name__ == "__main__":
    monte_carlo()
