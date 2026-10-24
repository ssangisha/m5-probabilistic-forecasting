import numpy as np


def mae(actual, forecast):
    """Average size of the error, in units sold."""
    return np.mean(np.abs(actual - forecast))


def wape(actual, forecast):
    """Total absolute error divided by total actual sales."""
    return np.sum(np.abs(actual - forecast)) / np.sum(actual)


def pinball_loss(actual, forecast, tau):
    """Quantile loss. tau is the quantile, for example 0.05, 0.5 or 0.95."""
    error = actual - forecast
    loss = np.where(error >= 0, tau * error, (tau - 1) * error)
    return np.mean(loss)


def coverage(actual, lower, upper):
    """Share of actual values that fall inside the interval."""
    inside = (actual >= lower) & (actual <= upper)
    return np.mean(inside)


def interval_width(lower, upper):
    """Average width of the interval. Wider intervals cover more, so report this too."""
    return np.mean(upper - lower)