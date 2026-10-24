import numpy as np
from metrics import mae, wape, pinball_loss, coverage

actual = np.array([0, 2, 4])
forecast = np.array([1, 2, 1])

# Absolute errors are 1, 0, 3, so MAE = 4 / 3
assert abs(mae(actual, forecast) - 4 / 3) < 1e-9

# WAPE = total absolute error / total actual = 4 / 6
assert abs(wape(actual, forecast) - 4 / 6) < 1e-9

# Pinball at tau = 0.9: losses are 0.1, 0, 2.7, so the mean is 2.8 / 3
assert abs(pinball_loss(actual, forecast, 0.9) - 2.8 / 3) < 1e-9

# Pinball at tau = 0.1: losses are 0.9, 0, 0.3, so the mean is 1.2 / 3
assert abs(pinball_loss(actual, forecast, 0.1) - 1.2 / 3) < 1e-9

# Coverage: the first two actuals are inside their intervals, the third is not
lower = np.array([0, 1, 2])
upper = np.array([1, 3, 3])
assert abs(coverage(actual, lower, upper) - 2 / 3) < 1e-9

print("All metric tests passed")