# BoN2024
Repository containing the solution of the Water Futures team for the Battle of the Network Demand Forecasting, held in Ferarra between 4th and 7th of July 2024.

Some (perhaps obvious) ideas (PS):
1. Using logarithms improves things.
2. DMA F can be forecasted sufficiently well via simple AR models (so a feature for a more advanced model could be to use those forecasts as a feature).
3. When using features among DMAs (for example the minimum value across DMAs for each day), use some kind of normalization before (because for example, this feature will tend to result in the value of DMA F or A).
4. Some interesting features: days_since_rain, real_feel, dew_point, heat_index, ewm of DMAs.
5. In the evaluation framework, in addition to the average scores, we should also evaluate different models with respect to their ranking (because model A can have a better average score than Model B, but it could be higher ranked).
6. Instead of forecasting the value of some DMAs directly, it is beneficial to first add some random noise, forecast that value and then take the average.
7. For those (year, week) there are missing values for at least 24 hours for at least one DMA: (2021, 14),(2021, 31),(2021, 32),(2021, 33),(2021, 34),(2021, 36),(2021, 49),(2022, 5),(2022, 6),(2022, 28)
8. Some financial technical indicators that seem to work: bollinger bands and rsi.
9. A nice of the self package that works well for outlier detection: SeasonalAD from adtk.detector.
10. Removing seasonal patterns and then forecasting the residuals, didn't work for me.
11. DMAs with strong time series properties: F and C.
12. DMAs with not strong time series properties: E, H and I.
