# inventory_policy.py
# Simulates a simple daily replenishment policy over one forecast window.

import numpy as np


def simulate_base_stock(demand, order_up_to):
    """Daily replenishment with overnight delivery and lost sales.

    demand and order_up_to have shape (number of series, number of days).
    Each morning stock is topped up to the order-up-to level. If yesterday's
    leftover stock is already higher, nothing is ordered. Unmet demand is lost.
    """
    n_series, n_days = demand.shape
    left_over = np.zeros(n_series)
    units_sold = np.zeros(n_series)
    units_lost = np.zeros(n_series)
    inventory_units_nights = np.zeros(n_series)   # sum of end-of-day stock
    stockout_days = np.zeros(n_series)

    for day in range(n_days):
        stock = np.maximum(order_up_to[:, day], left_over)
        sold = np.minimum(stock, demand[:, day])
        lost = demand[:, day] - sold
        left_over = stock - sold

        units_sold += sold
        units_lost += lost
        inventory_units_nights += left_over
        stockout_days += (lost > 0)

    return {
        "units_sold": units_sold,
        "units_lost": units_lost,
        "inventory_units_nights": inventory_units_nights,
        "stockout_days": stockout_days,
    }