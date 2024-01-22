import re
import numpy as np

from pathlib import Path

MC_DIR_NAME = "20240122-124908_VT_MonteCarlo_daytona_500_sdx_1s_loop_10s_50Hz"


DATA_PATH = Path(__file__).parents[1] / "data"
MC_PATH = DATA_PATH / "monte_carlos" / MC_DIR_NAME
FIGURES_PATH = DATA_PATH / "figures" / f"MC_{MC_DIR_NAME}"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

for monte_carlo in MC_PATH.glob("*.npz"):
    current_js = int(re.findall(r"\d+", monte_carlo.stem)[0])
    data = np.load(monte_carlo)

    print("")
