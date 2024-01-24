import numpy as np
import navsim as ns
import simulate_vt as vt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from eleventh_hour.data import create_padded_df

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
    conf, meas_sim, corr_sim = vt.setup_simulation(disable_progress=DISABLE_PROGRESS)
    output_dir = create_sim_dir(conf=conf)

    mc_results = defaultdict(lambda: [])

    for js in JS:
        # set current js
        for constellation, properties in conf.constellations.emitters.items():
            if constellation in INTERFERED_CONSTELLATIONS:
                properties.js = js

        # perform monte carlos
        for _ in tqdm(range(NSIMS), desc=f"simulating with {js} dB J/S"):
            results, _ = vt.simulate(
                conf=conf,
                meas_sim=meas_sim,
                corr_sim=corr_sim,
                disable_progress=DISABLE_PROGRESS,
            )
            mc_results = process_sim_results(results=results, mc_results=mc_results)

            meas_sim.clear_observables()
            corr_sim.clear_errors()

        results = process_mc_results(time=results.states.time, mc_results=mc_results)
        np.savez_compressed(output_dir / f"js{int(js):02}", **results)


def process_sim_results(results: vt.SimulationResults, mc_results: dict):
    chip_error = create_padded_df(data=results.true_chip_error)
    prange_error = create_padded_df(data=results.true_prange_error)
    ferror = create_padded_df(data=results.true_ferror)

    mean_ptrack = np.mean(np.abs(chip_error.gps.to_numpy()[-1]) < 0.5)
    mean_chip_error = np.mean(np.abs(chip_error.gps.to_numpy()), axis=1)
    mean_prange_error = np.mean(np.abs(prange_error.gps.to_numpy()), axis=1)
    mean_ferror = np.mean(np.abs(ferror.gps.to_numpy()), axis=1)

    mc_results["ptrack"].append(mean_ptrack)
    mc_results["chip_error"].append(mean_chip_error)
    mc_results["prange_error"].append(mean_prange_error)
    mc_results["ferror"].append(mean_ferror)

    mc_results["pos_error"].append(results.errors.pos)
    mc_results["vel_error"].append(results.errors.vel)
    mc_results["cb_error"].append(results.errors.clock_bias)
    mc_results["cd_error"].append(results.errors.clock_drift)

    mc_results["pos_cov"].append(results.covariances.pos)
    mc_results["vel_cov"].append(results.covariances.vel)
    mc_results["cb_cov"].append(results.covariances.clock_bias)
    mc_results["cd_cov"].append(results.covariances.clock_drift)

    return mc_results


def process_mc_results(time: np.ndarray, mc_results: dict):
    results = defaultdict()
    results["time"] = time

    for key, value in mc_results.items():
        new_key = f"mean_{key}"
        mean_value = np.mean(value, axis=0)
        results[new_key] = mean_value

        new_key = f"var_{key}"
        var_value = np.var(value, axis=0)
        results[new_key] = var_value

        if "error" in key:
            new_key = f"rms_{key}"
            value = np.array(value)
            rms_value = np.sqrt(np.mean(value**2, axis=-1))
            results[new_key] = rms_value

    return dict(results)


def create_sim_dir(conf: ns.SignalConfiguration):
    now = datetime.now().strftime(format="%Y%m%d-%H%M%S")

    dir_name = f"{now}_VT_MonteCarlo_{vt.TRAJECTORY}_{int(conf.time.duration)}s_{int(conf.time.fsim)}Hz"
    sim_dir = MC_PATH / dir_name

    sim_dir.mkdir(parents=True, exist_ok=True)

    return sim_dir


if __name__ == "__main__":
    monte_carlo()
