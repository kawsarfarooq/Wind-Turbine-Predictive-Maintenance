# Cost-aware maintenance findings

The benchmark contains 62 held-out failure episodes from seven turbines under
each of three independent data seeds (21 turbine/data clusters). PTO and the
residual threshold are tuned only on training episodes; DFL is repeated with
three policy seeds.

The oracle lower bound costs 50.5 k EUR/episode. Among feasible learned
policies, **PTO is strongest** at 89.0 k EUR (clustered 95% interval
78.1-100.0), with 3.2% failures and 38.6 k EUR oracle regret. DFL costs 146.6 k
EUR and has 16.7% failures; it does not outperform the fairly calibrated PTO
baseline. Reactive maintenance costs 300.0 k EUR and fails in every episode.
These are controlled synthetic costs, not operator-calibrated economic
estimates.
