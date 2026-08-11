import matplotlib.pyplot as plt
import numpy as np
import market as p
from simulate import run_sim

def main():
    seed = 10
    np.random.seed(seed)

    lam = np.random.lognormal(mean=np.log(0.2), sigma=0.5)
    prices = p.stock_path(p.S_0, p.r, p.sigma, p.dt, p.num_steps, p.jump_prob, p.jump_mean, p.jump_std)

    pnl_log, inv_log, delta_log, gamma_log = run_sim(prices, lam, p)

    print(f"Final PnL: ${pnl_log[-1]:.2f}")
    print(f"Final Delta: {delta_log[-1]:.4f}")
    print(f"Final Gamma: {gamma_log[-1]:.4f}")

    print("Final Inventory:")
    for (instrument, K, T0), log in inv_log.items():
        print(f"  {instrument} K={K} T={T0}: {log[-1]:.0f} contracts")

    plot_results(pnl_log, inv_log, delta_log, gamma_log, prices, p.strikes, seed)

def plot_results(pnl_log, inv_log, delta_log, gamma_log, prices, strikes, seed):
    expiries = sorted(set(T0 for K, T0 in strikes))
    strike_vals = sorted(set(K for K, T0 in strikes))
    n_exp_rows = len(expiries)
    n_strike_cols = len(strike_vals)

    total_rows = 2 + n_exp_rows + 2
    fig = plt.figure(figsize=(4 * n_strike_cols, 2.2 * total_rows), constrained_layout=True)
    gs = fig.add_gridspec(total_rows, n_strike_cols)

    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(prices, color='tab:blue')
    ax_price.set_title('Stock Price')
    ax_price.set_ylabel('Price ($)')
    ax_price.grid(alpha=0.4)

    ax_pnl = fig.add_subplot(gs[1, :], sharex=ax_price)
    ax_pnl.plot(pnl_log, color='tab:green')
    ax_pnl.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_pnl.set_title('PnL')
    ax_pnl.set_ylabel('PnL ($)')
    ax_pnl.grid(alpha=0.4)

    for r, T0 in enumerate(expiries):
        for c, K in enumerate(strike_vals):
            ax = fig.add_subplot(gs[2 + r, c], sharex=ax_price)
            ax.plot(inv_log[('call', K, T0)], label='call', color='tab:orange')
            ax.plot(inv_log[('put', K, T0)], label='put', color='tab:purple')
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_title(f'K={K}, T={T0}', fontsize=10)
            ax.grid(alpha=0.4)
            if c == 0:
                ax.set_ylabel('Contracts')
            if r == 0 and c == 0:
                ax.legend(loc='upper right', fontsize=8)

    ax_delta = fig.add_subplot(gs[2 + n_exp_rows, :], sharex=ax_price)
    ax_delta.plot(delta_log, color='tab:red')
    ax_delta.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_delta.set_title('Net Delta Exposure')
    ax_delta.set_ylabel('Delta')
    ax_delta.grid(alpha=0.4)

    ax_gamma = fig.add_subplot(gs[3 + n_exp_rows, :], sharex=ax_price)
    ax_gamma.plot(gamma_log, color='tab:brown')
    ax_gamma.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_gamma.set_title('Net Gamma Exposure')
    ax_gamma.set_ylabel('Gamma')
    ax_gamma.set_xlabel('Time (seconds)')
    ax_gamma.grid(alpha=0.4)

    plt.savefig(f'results/seed_{seed}_run.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()