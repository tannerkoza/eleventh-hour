import numpy as np
import navsim as ns
import efficiency as dpe

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from eleventh_hour.navigators import create_padded_df
from efficiency import NTOTAL, NLEO

# sim parameters
NSIMS = 200
RESOLUTION = np.arange(0.075, 0.01, -0.005, dtype=float)
DISABLE_PROGRESS = True
# path
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "conf"
DATA_PATH = PROJECT_PATH / "data"
MC_PATH = DATA_PATH / "monte_carlos"


def monte_carlo():
    conf, meas_sim, corr_sim = dpe.setup_simulation(disable_progress=DISABLE_PROGRESS)
    output_dir = create_sim_dir(conf=conf)

    mc_results = defaultdict(lambda: [])

    for resolution in RESOLUTION:
        # perform monte carlos
        for _ in tqdm(range(NSIMS), desc=f"simulating with {resolution:03} resolution"):

            results, particles = dpe.simulate(
                conf=conf,
                meas_sim=meas_sim,
                corr_sim=corr_sim,
                resolution=resolution,
                disable_progress=DISABLE_PROGRESS,
            )

            mc_results = process_sim_results(results=results, mc_results=mc_results)

            meas_sim.clear_observables()
            corr_sim.clear_errors()

        results = process_mc_results(time=results.states.time, mc_results=mc_results)
        mc_results = defaultdict(lambda: [])

        np.savez_compressed(output_dir / f"{int(particles):04}", **results)


def process_sim_results(results: dpe.SimulationResults, mc_results: dict):
    chip_error = create_padded_df(data=results.chip_errors)
    ferror = create_padded_df(data=results.ferrors)
    prange_error = create_padded_df(data=results.prange_errors)
    prange_rate_error = create_padded_df(data=results.prange_rate_errors)

    chip_error = np.ma.array(
        chip_error.gps.to_numpy(), mask=np.isnan(chip_error.gps.to_numpy())
    )
    ferror = np.ma.array(ferror.gps.to_numpy(), mask=np.isnan(ferror.gps.to_numpy()))
    prange_error = np.ma.array(
        prange_error.gps.to_numpy(), mask=np.isnan(prange_error.gps.to_numpy())
    )
    prange_rate_error = np.ma.array(
        prange_rate_error.gps.to_numpy(),
        mask=np.isnan(prange_rate_error.gps.to_numpy()),
    )

    mean_ptrack = np.mean(np.abs(chip_error[-1]) < 0.5)
    mean_chip_error = np.mean(np.abs(chip_error), axis=1)
    mean_ferror = np.mean(np.abs(ferror), axis=1)
    mean_prange_error = np.mean(np.abs(prange_error), axis=1)
    mean_prange_rate_error = np.mean(np.abs(prange_rate_error), axis=1)

    mc_results["ptrack"].append(mean_ptrack)
    mc_results["chip_error"].append(mean_chip_error)
    mc_results["ferror"].append(mean_ferror)
    mc_results["prange_error"].append(mean_prange_error)
    mc_results["prange_rate_error"].append(mean_prange_rate_error)

    # mc_results["pos"].append(results.states.enu_pos)
    # mc_results["vel"].append(results.states.enu_vel)
    # mc_results["cb"].append(results.states.clock_bias)
    # mc_results["cd"].append(results.states.clock_drift)

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
    results["pos_error"] = mc_results["pos_error"]
    results["vel_error"] = mc_results["vel_error"]
    results["cb_error"] = mc_results["cb_error"]
    results["cd_error"] = mc_results["cd_error"]

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

            new_key = f"final_{key}"
            value = np.array(value).T[-1]
            results[new_key] = value

    return dict(results)


def create_sim_dir(conf: ns.SignalConfiguration):
    now = datetime.now().strftime(format="%Y%m%d-%H%M%S")

    traj = dpe.TRAJECTORY
    if dpe.IS_STATIC:
        mp = "STATIC"
    else:
        mp = "DYNAMIC"

    dir_name = f"{now}{NTOTAL}SV_{NLEO}LEO_DPE_Efficiency_MonteCarlo_{traj}_{mp}_{int(conf.time.duration)}s_{int(conf.time.fsim)}Hz"
    sim_dir = MC_PATH / dir_name

    sim_dir.mkdir(parents=True, exist_ok=True)

    return sim_dir


if __name__ == "__main__":
    monte_carlo()
