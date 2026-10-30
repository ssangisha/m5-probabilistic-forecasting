# Inventory simulation: assumptions and limits

Simulation results under stated assumptions. Costs are in relative units, not
money. Nothing here is a claim about savings achieved in a real business.

## Setup
- Replenishment: daily review. Each morning, stock is topped up to an
  order-up-to level. If leftover stock is already higher, nothing is ordered.
- Lead time: overnight. Orders arrive before the store opens.
- Starting stock: zero on the shelf, filled to the day-1 level before day 1.
- Forecast timing: quantile forecasts are made once at the start of each
  28-day window and not updated during the window.
- Unmet demand is lost. There are no backorders.
- Stock levels are whole units (rounded to the nearest unit; rounding up or
  down was tested and did not change the ranking).
- Leftover stock carries over to the next day. Nothing expires.

## Costs (relative units)
- Holding cost: 1 per unit left on the shelf overnight.
- Shortage cost: 9, 19 or 39 per lost unit of demand.
- Target quantile = shortage cost / (shortage cost + holding cost), which gives
  0.90, 0.95 and 0.975 (the newsvendor critical fraction).
- All items are valued equally. There is no price weighting.

## Policies compared
1. Seasonal-naive, no buffer: stock equals the same weekday last week.
2. Textbook rule: mean plus z times the standard deviation of the last 8 same
   weekdays (z from the normal distribution at the target quantile).
3. Empirical-quantile baseline: quantile of the last 8 same weekdays.
4. LightGBM (tuned) quantile forecast at the target quantile.
5. N-HiTS quantile forecast at the target quantile.

## Not modelled
Order costs, minimum order sizes, shelf life, shelf capacity, lead times longer
than one day, price weighting, promotions beyond what the models see, and new
products without history.

## Known limits
- 720 series stratified from the M5 dataset, not the whole dataset.
- Four 28-day windows (training ends 2015-06-19, 2015-09-18, 2015-12-18,
  2016-03-18). Fold 4 was not used for any tuning.
- Because leftover stock carries over, realized stockout rates are below the
  target levels, so absolute costs are not the minimum possible. The comparison
  between policies is like for like.
- Bootstrap intervals cover which series were sampled, not model training
  randomness. A separate seed check covers N-HiTS training randomness.