import numpy as np
from reg_from_scratch import ols

# mock data
N = 100
x = np.random.uniform(-10, 10, (N, 2))
u = np.random.normal(0, 1, N)
a = 3
b1 = 5
b2 = -3
y = a + b1 * x[:, 0] + b2 * x[:, 1] + u

ols(x, y)