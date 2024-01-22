import numpy as np

a = np.atleast_3d(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
print(a.shape)
est = np.array([1, 2, 3, 4])

b = a - est
cov = np.outer(a, a)
print(cov)
