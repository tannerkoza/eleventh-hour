import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path

# user-defined variables
MC_DIR_NAME = "daytonafr"
RE_EXPR = "gps|iridium|vp|dpe"
TITLES = ["East", "North", "Up"]
CONTEXT = "paper"
MARKER_SIZE = 10
CMAP = "viridis"
CN0_AXIS = "Nominal $C/N_{0}$ Attenuation [dB]"

# path creation
DATA_PATH = Path(__file__).parents[1] / "data"
MC_PATH = DATA_PATH / "monte_carlos" / MC_DIR_NAME
FIGURES_PATH = DATA_PATH / "figures" / f"MC_{MC_DIR_NAME}"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    data = defaultdict(lambda: defaultdict(lambda: []))

    for scenario in MC_PATH.glob("*"):
        name = scenario.name.casefold()
        key = "-".join(re.findall(RE_EXPR, name))

        for monte_carlo in scenario.glob("*.npz"):
            current_js = int(re.findall(r"\d+", monte_carlo.stem)[0])
            data[key][current_js] = np.load(monte_carlo)

    return data


def plot():
    sns.set_context(CONTEXT)
    # sns.set_palette(CMAP)

    scenarios = load_data()

    ptrack_fig, ptrack_ax = plt.subplots()
    rmspe_fig, rmspe_ax = plt.subplots()
    rmsve_fig, rmsve_ax = plt.subplots()
    # rmspe_fig, rmspe_ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    # rmsve_fig, rmsve_ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    rmscbe_fig, rmscbe_ax = plt.subplots()
    rmscde_fig, rmscde_ax = plt.subplots()

    for name, data in scenarios.items():
        js = -np.fromiter(data.keys(), dtype=float)
        sorted_js = np.sort(js)
        sorted_js_indices = np.argsort(js)

        ptrack = np.array([js["mean_ptrack"] for js in data.values()]) * 100.0
        pos = np.array([js["pos"] for js in data.values()])
        rmspe = np.array([js["rms_pos_error"] for js in data.values()]).mean(axis=1).T
        rmsve = np.array([js["rms_vel_error"] for js in data.values()]).mean(axis=1).T
        rmscbe = np.array([js["rms_cb_error"] for js in data.values()]).mean(axis=1)
        rmscde = np.array([js["rms_cd_error"] for js in data.values()]).mean(axis=1)

        ptrack_ax.plot(
            sorted_js,
            ptrack[sorted_js_indices],
            marker="*",
            label=name,
            markersize=MARKER_SIZE,
        )

        # for index, (title, ax) in enumerate(zip(TITLES, rmspe_ax)):
        #     ax.plot(js, rmspe[index], "*", label=name, markersize=MARKER_SIZE)
        #     ax.set_title(title)

        # for index, (title, ax) in enumerate(zip(TITLES, rmsve_ax)):
        #     ax.plot(js, rmsve[index], "*", label=name, markersize=MARKER_SIZE)
        #     ax.set_title(title)

        rmspe_ax.set_yscale("log")
        rmspe_ax.plot(
            sorted_js,
            np.linalg.norm(rmspe, axis=0)[sorted_js_indices],
            marker="*",
            label=name,
            markersize=MARKER_SIZE,
        )

        rmsve_ax.set_yscale("log")
        rmsve_ax.plot(
            sorted_js,
            np.linalg.norm(rmsve, axis=0)[sorted_js_indices],
            marker="*",
            label=name,
            markersize=MARKER_SIZE,
        )

        rmscbe_ax.set_yscale("log")
        rmscbe_ax.plot(
            sorted_js,
            rmscbe[sorted_js_indices],
            marker="*",
            label=name,
            markersize=MARKER_SIZE,
        )

        rmscde_ax.set_yscale("log")
        rmscde_ax.plot(
            sorted_js,
            rmscde[sorted_js_indices],
            marker="*",
            label=name,
            markersize=MARKER_SIZE,
        )

    ptrack_fig.supxlabel(CN0_AXIS)
    ptrack_fig.supylabel("Probability of Tracking [%]")
    ptrack_ax.legend()
    ptrack_fig.tight_layout()

    rmspe_fig.supxlabel(CN0_AXIS)
    rmspe_fig.supylabel("RMS Position Error [m]")
    handles, labels = rmspe_ax.get_legend_handles_labels()
    rmspe_ax.legend(handles, labels)
    rmspe_fig.tight_layout()

    rmsve_fig.supxlabel(CN0_AXIS)
    rmsve_fig.supylabel("RMS Velocity Error [m/s]")
    handles, labels = rmsve_ax.get_legend_handles_labels()
    rmsve_ax.legend(handles, labels)
    rmsve_fig.tight_layout()

    rmscbe_fig.supxlabel(CN0_AXIS)
    rmscbe_fig.supylabel("RMS Clock Bias Error [m]")
    rmscbe_ax.legend()
    rmscbe_fig.tight_layout()

    rmscde_fig.supxlabel(CN0_AXIS)
    rmscde_fig.supylabel("RMS Clock Drift Error [m/s]")
    rmscde_ax.legend()
    rmscde_fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot()
