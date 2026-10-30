# test_inventory_policy.py
# Checks the simulation against small examples worked out by hand.

import numpy as np
from inventory_policy import simulate_base_stock

# Demand 2, 0, 3 with order-up-to level 2 every day:
#   day 1: stock 2, sell 2, left 0
#   day 2: stock 2, sell 0, left 2
#   day 3: stock 2, sell 2, lose 1, left 0
result = simulate_base_stock(np.array([[2, 0, 3]]), np.array([[2, 2, 2]]))
assert result["units_sold"][0] == 4
assert result["units_lost"][0] == 1
assert result["inventory_units_nights"][0] == 2
assert result["stockout_days"][0] == 1

# Leftover stock must not be thrown away when the new level is lower:
#   day 1: stock 3, sell 1, left 2
#   day 2: level is 0 but 2 are left, so stock 2, sell 1, left 1
result = simulate_base_stock(np.array([[1, 1]]), np.array([[3, 0]]))
assert result["units_sold"][0] == 2
assert result["units_lost"][0] == 0
assert result["inventory_units_nights"][0] == 3

print("All inventory policy tests passed")