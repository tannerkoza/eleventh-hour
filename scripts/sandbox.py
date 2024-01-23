import numpy as np
from itertools import product

state = np.array([1, 0])
P = np.array([[-5, 0, 5], [-10, 0, 10]]).T


b = state + P
# a = np.array(list(product(*b))).T

print(b)
