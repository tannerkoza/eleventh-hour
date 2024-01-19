import numpy as np
from navtools.signals import bpsk_correlator

a = np.array([20,23, 25,56,75])
b = np.array([18,23, 25,56,75])

c = np.where(a < 22, b, a)
print(c)