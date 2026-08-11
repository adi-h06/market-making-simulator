import numpy as np
from pricing import black_scholes_call, black_scholes_put, delta_call, delta_put, gamma_bs

def run_sim(prices, lam, p):
    num_steps = len(prices)
    cash = 0

    K_arr = np.array([K for K, T0 in p.strikes])
    T0_arr = np.array([T0 for K, T0 in p.strikes])
    sigma_arr = p.sigma + p.skew_slope * (K_arr - p.S_0) / p.S_0
    n = len(p.strikes)
    keys_call = [('call', K, T0) for K, T0 in p.strikes]
    keys_put = [('put', K, T0) for K, T0 in p.strikes]

    inv_call = np.zeros(n)
    inv_put = np.zeros(n)
    hedge_shares = 0
    pending_delta_hedges = []
    spread_pnl = 0

    jump_events = precompute_jump_events(prices, p, n)

    pnl_log = np.zeros(num_steps)
    inv_log = {key: np.zeros(num_steps) for key in keys_call + keys_put}
    delta_log = np.zeros(num_steps)
    gamma_log = np.zeros(num_steps)

    for i in range(num_steps):
        S = prices[i]
        T_rem = np.maximum(T0_arr - i * p.dt, 1e-6)

        fair_call = black_scholes_call(S, K_arr, T_rem, p.r, sigma_arr)
        delta_c = delta_call(S, K_arr, T_rem, p.r, sigma_arr)
        fair_put = black_scholes_put(S, K_arr, T_rem, p.r, sigma_arr)
        delta_p = delta_put(S, K_arr, T_rem, p.r, sigma_arr)
        gamma_arr = gamma_bs(S, K_arr, T_rem, p.r, sigma_arr)

        eff_spread_call = p.spread + p.inventory_penalty * np.abs(inv_call)
        center_call = fair_call - p.skew_sens * inv_call
        bid_call, ask_call = center_call - eff_spread_call / 2, center_call + eff_spread_call / 2

        eff_spread_put = p.spread + p.inventory_penalty * np.abs(inv_put)
        center_put = fair_put - p.skew_sens * inv_put
        bid_put, ask_put = center_put - eff_spread_put / 2, center_put + eff_spread_put / 2

        cash_before = cash
        cash, inv_call, inv_put = order_flow(i, prices, fair_call, bid_call, ask_call,
                                             fair_put, bid_put, ask_put, cash, inv_call, inv_put, n, p, lam)

        if i in jump_events:
            for idx, instrument, side in jump_events[i]:
                if instrument == 'call':
                    distance = (ask_call[idx] - fair_call[idx]) if side == 'buy' else (fair_call[idx] - bid_call[idx])
                    cnt = np.random.poisson(p.jump_informed_strength * np.exp(-p.jump_price_sens * distance))
                    if side == 'buy':
                        cnt = max(min(cnt, inv_call[idx] + p.max_inventory), 0)
                        cash += ask_call[idx] * cnt; inv_call[idx] -= cnt
                    else:
                        cnt = max(min(cnt, p.max_inventory - inv_call[idx]), 0)
                        cash -= bid_call[idx] * cnt; inv_call[idx] += cnt
                else:
                    distance = (ask_put[idx] - fair_put[idx]) if side == 'buy' else (fair_put[idx] - bid_put[idx])
                    cnt = np.random.poisson(p.jump_informed_strength * np.exp(-p.jump_price_sens * distance))
                    if side == 'buy':
                        cnt = max(min(cnt, inv_put[idx] + p.max_inventory), 0)
                        cash += ask_put[idx] * cnt; inv_put[idx] -= cnt
                    else:
                        cnt = max(min(cnt, p.max_inventory - inv_put[idx]), 0)
                        cash -= bid_put[idx] * cnt; inv_put[idx] += cnt

        spread_pnl += (cash - cash_before)

        book_delta = np.sum(inv_call * delta_c) + np.sum(inv_put * delta_p)
        book_gamma = np.sum(inv_call * gamma_arr) + np.sum(inv_put * gamma_arr)

        still_pending = []
        executed_stock = 0
        for exec_idx, amount in pending_delta_hedges:
            if exec_idx <= i:
                hedge_shares += amount
                executed_stock += amount
            else:
                still_pending.append((exec_idx, amount))
        pending_delta_hedges = still_pending
        cash -= executed_stock * S

        net_delta = book_delta + hedge_shares
        projected_delta = net_delta + sum(a for _, a in pending_delta_hedges)
        if abs(projected_delta) > p.hedge_threshold:
            pending_delta_hedges.append((i + p.hedge_lag, -projected_delta))

        current_pnl = cash + np.sum(inv_call * fair_call) + np.sum(inv_put * fair_put) + hedge_shares * S
        pnl_log[i] = current_pnl
        for idx, key in enumerate(keys_call):
            inv_log[key][i] = inv_call[idx]
        for idx, key in enumerate(keys_put):
            inv_log[key][i] = inv_put[idx]
        delta_log[i] = net_delta
        gamma_log[i] = book_gamma

    print(f"Spread PnL: ${spread_pnl:.2f}")
    print(f"Jump events: {len(jump_events)}")
    return pnl_log, inv_log, delta_log, gamma_log

def order_flow(i, prices, fair_call, bid_call, ask_call, fair_put, bid_put, ask_put, cash, inv_call, inv_put, n, p, lam):
    lam_each = lam / (2 * n)
    price_up = prices[min(i + p.horizon, len(prices) - 1)] > prices[i]
    correct_side_call = 'buy' if price_up else 'sell'
    correct_side_put = 'sell' if price_up else 'buy'

    cash, inv_call = process_side(lam_each, fair_call, bid_call, ask_call, cash, inv_call, correct_side_call, n, p)
    cash, inv_put = process_side(lam_each, fair_put, bid_put, ask_put, cash, inv_put, correct_side_put, n, p)
    return cash, inv_call, inv_put

def process_side(lam_each, fair, bid, ask, cash, inv, correct_side, n, p):
    n_buy = np.random.poisson(lam_each * np.exp(-p.spread_sens * (ask - fair)))
    n_sell = np.random.poisson(lam_each * np.exp(-p.spread_sens * (fair - bid)))

    for idx in range(n):
        for default_side, count in (('buy', n_buy[idx]), ('sell', n_sell[idx])):
            for _ in range(count):
                side = resolve_side(default_side, correct_side, p)
                if side == 'buy' and inv[idx] > -p.max_inventory:
                    cash += ask[idx]; inv[idx] -= 1
                elif side == 'sell' and inv[idx] < p.max_inventory:
                    cash -= bid[idx]; inv[idx] += 1
    return cash, inv

def resolve_side(default_side, correct_side, p):
    if np.random.random() < p.informed_ratio:
        return correct_side if np.random.random() < p.edge else default_side
    return default_side

def precompute_jump_events(prices, p, n):
    log_rets = np.diff(np.log(prices))
    jump_indices = np.where(np.abs(log_rets) > p.jump_signal_threshold)[0] + 1
    events = {}
    for j in jump_indices:
        trigger = max(j - np.random.randint(p.jump_lead_min, p.jump_lead_max + 1), 0)
        price_up = log_rets[j - 1] > 0
        n_legs = min(np.random.randint(p.jump_legs_min, p.jump_legs_max + 1), n)
        chosen_idx = np.random.choice(n, size=n_legs, replace=False)
        legs = []
        for idx in chosen_idx:
            instrument = np.random.choice(['call', 'put'])
            side = ('buy' if price_up else 'sell') if instrument == 'call' else ('sell' if price_up else 'buy')
            legs.append((idx, instrument, side))
        events[trigger] = legs
    return events