import numpy as np
from metrics import wape, pinball_loss, coverage, interval_width

TAUS = [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975]


def score(result):
    """Score one model's quantile forecasts on one fold."""
    actual = result["sales"]
    pinballs = [pinball_loss(actual, result[f"q{tau}"], tau) for tau in TAUS]
    return {
        "mean_pinball": np.mean(pinballs),
        "wape_median": wape(actual, result["q0.5"]),
        "cov_80": coverage(actual, result["q0.1"], result["q0.9"]),
        "cov_90": coverage(actual, result["q0.05"], result["q0.95"]),
        "cov_95": coverage(actual, result["q0.025"], result["q0.975"]),
        "width_90": interval_width(result["q0.05"], result["q0.95"]),
    }