import numpy as np

S_0 = 100.0
strikes = [(95.0, 0.25), (100.0, 0.25), (105.0, 0.25),
           (95.0, 0.5), (100.0, 0.5), (105.0, 0.5)]
r = 0.05
sigma = 0.35
jump_prob = 0.000003
jump_mean = -0.01
jump_std = 0.04

dt = 1 / 252 / 23400
num_steps = 23400

spread = 0.05
informed_ratio = 0.3
edge = 0.65
horizon = 60

hedge_threshold = 10
hedge_lag = 2
inventory_penalty = 0.0005
spread_sens = 40.0
skew_sens = 0.00125
max_inventory = 50

jump_lead_min = 0
jump_lead_max = 4
jump_signal_threshold = 0.015
jump_informed_strength = 25
jump_price_sens = 5
jump_legs_min = 1
jump_legs_max = 2

skew_slope = -0.5


def stock_path(S_0, r, sigma, dt, num_steps, jump_prob, jump_mean, jump_std):
    Z = np.random.normal(size=num_steps)
    returns = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    jumps = np.random.random(size=num_steps) < jump_prob
    jump_sizes = np.exp(np.random.normal(jump_mean, jump_std, size=num_steps))
    jump_multi = np.where(jumps, jump_sizes, 1.0)
    return S_0 * np.cumprod(returns * jump_multi)