import numpy as np
import navsim as ns
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from warnings import simplefilter
from simulate_vt import setup_simulation, simulate, TRAJECTORY

# sim parameters
NSIMS = 100
JS = np.arange(0, 45, 5, dtype=float)
INTERFERED_CONSTELLATIONS = ["gps"]
DISABLE_PROGRESS = True

# path
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
MC_PATH = DATA_PATH / "monte_carlos"


def monte_carlo():
    conf, meas_sim, corr_sim = setup_simulation(disable_progress=DISABLE_PROGRESS)
    mc_dir = create_mc_dir(conf=conf)

    for js in JS:
        # set current js
        for constellation, properties in conf.constellations.emitters.items():
            if constellation in INTERFERED_CONSTELLATIONS:
                properties.js = js

        js_path = mc_dir / f"js{int(js)}"
        js_path.mkdir(parents=True, exist_ok=True)

        # perform monte carlos
        for sim in tqdm(range(NSIMS), desc=f"simulating with {js} dB J/S"):
            rx_truth_states, vdfll = simulate(
                conf=conf,
                meas_sim=meas_sim,
                corr_sim=corr_sim,
                disable_progress=DISABLE_PROGRESS,
            )
            data = {
                "rx_truth_states": pd.DataFrame([rx_truth_states]),
                "vdfll_rx_states": vdfll.rx_states,
                "vdfll_code_errors": vdfll.code_errors,
                "vdfll_ferrors": vdfll.ferrors,
                "vdfll_prange_errors": vdfll.prange_errors,
                "vdfll_prange_rate_errors": vdfll.prange_rate_errors,
            }

            output_path = js_path / f"sim{sim}"
            save_data(output_path=output_path, data=data)

            meas_sim.clear_observables()


def create_mc_dir(conf: ns.SignalConfiguration):
    now = datetime.now().strftime(format="%Y%m%d-%H%M%S")

    dir_name = f"{now}_VT_MonteCarlo_{TRAJECTORY}_{int(conf.time.duration)}s_{int(conf.time.fsim)}Hz"
    mc_dir = MC_PATH / dir_name

    mc_dir.mkdir(parents=True, exist_ok=True)

    return mc_dir


def save_data(output_path: Path, data: dict):
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    for name, df in data.items():
        df.to_hdf(output_path.with_suffix(".h5"), key=name, mode="a")


if __name__ == "__main__":
    monte_carlo()
