import numpy as np
from numba import njit
from itertools import product
from navtools.conversions import ecef2lla, enu2ecef

epoch_emitters = ["G10", "G12"]
emitters = ["G10", "G14"]
cn0 = np.array([45, 46])

new_emitters = set(epoch_emitters) - set(emitters)
removed_emitters = set(emitters) - set(epoch_emitters)

removed_indices = [emitters.index(emitter) for emitter in removed_emitters]

cn0 = np.delete(cn0, [])

print()
