"""
LNG cargo hedging with Monte Carlo
==================================

Question
--------
An LNG trader buys a cargo at a fixed cost and sells it in 3 months at a market
price (JKM, the Asian benchmark). How uncertain is the margin, and how much does
a hedge with TTF derivatives (the European benchmark) reduce that uncertainty?

Method
------------------
* Prices: one-factor mean-reverting log-price model (Schwartz, 1997), consistent
  with the forward curve: E[S_t] = F0. JKM and TTF are correlated.
* Monte Carlo: n_paths simulated daily price paths; the cargo is priced on the
  average of the last `window` days.
* Strategies: no hedge, swap with h = 1, swap with the minimum-variance ratio h*,
  and a put option on TTF.
* Risk metrics: mean, std, P5, Expected Shortfall 5%, P95, probability of loss.

Notes
-----
* Market parameters are ILLUSTRATIVE (order of magnitude only).
  Calibrate them on real JKM/TTF data before drawing real conclusions.
* Run:  python lng_hedge.py   (prints tables, then opens two plot windows)
"""
from dataclasses import dataclass, replace
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np

# Strategy names.
NO_HEDGE = "No hedge"
SWAP_FULL = "Swap TTF h=1"
SWAP_MINVAR = "Swap TTF h*"


# 1. INPUT ==============================
@dataclass
class Params:
    """
    Change a value here to run a different scenario.
    """
    # --- Cargo ---
    volume: float = 3.5e6      # [MMBtu] energy in one cargo (~170k m3 carrier, net of heel and boil-off)
    cost: float = 9.5          # [$/MMBtu] all-in cost (purchase + liquefaction + shipping)
    # --- Market (illustrative) ---
    F0_jkm: float = 12.0       # [$/MMBtu] JKM forward for the delivery month (expected sale price)
    F0_ttf: float = 11.0       # [$/MMBtu] TTF forward for the same period (price of the hedge instrument)
    sigma_jkm: float = 0.60    # [1/sqrt(year)] volatility of the JKM factor: how agitated the price is
    sigma_ttf: float = 0.65    # [1/sqrt(year)] volatility of the TTF factor
    kappa: float = 2.0         # [1/year] mean-reversion speed: strength of the "leash" (half-life ~ 4 months)
    rho: float = 0.85          # [-] correlation between JKM and TTF random shocks (0 = unrelated, 1 = identical)
    # --- Timing ---
    T: float = 0.25            # [year] time from today to delivery (3 months)
    n_steps: int = 63          # [trading days] number of daily steps in T
    window: int = 21           # [trading days] pricing window: sale price = average of the last 21 days
    # --- Option and Monte Carlo ---
    put_strike_ratio: float = 0.90   # [-] put strike as a fraction of the TTF forward
    n_paths: int = 100_000     # [-] number of simulated futures
    seed: int = 42             # [-] random seed: same seed, same results


# 2. MODEL ==============================
def simulate_prices(p):
    """
    Simulate correlated JKM and TTF daily price paths.

    For each index, X is a zero-mean Ornstein-Uhlenbeck process
    (dX = -kappa*X dt + sigma dW) simulated with its exact discretisation, and the
    price is S = F0 * exp(X - Var(X)/2), which guarantees E[S_t] = F0 (no drift
    away from the forward). The two indices share correlated random shocks (rho).

    Returns
    -------
    S_jkm, S_ttf : ndarray, shape (n_paths, n_steps), [$/MMBtu]
        One row per simulated future, one column per trading day.
    """
    rng = np.random.default_rng(p.seed)
    dt = p.T / p.n_steps                                   # [year] length of one step (1 trading day)
    t = np.arange(1, p.n_steps + 1) * dt                   # [year] time grid: dt, 2*dt, ..., T
    a = np.exp(-p.kappa * dt)                              # [-] share of the deviation left after one step
    step = np.sqrt((1 - a**2) / (2 * p.kappa))             # [-] exact std of one random kick (for sigma = 1)

    # Random numbers: one row per path, one column per day.
    z_jkm = rng.standard_normal((p.n_paths, p.n_steps))
    # TTF shocks = rho parts of the JKM shocks + sqrt(1 - rho^2) parts of fresh ones -> correlation = rho.
    z_ttf = p.rho * z_jkm + np.sqrt(1 - p.rho**2) * rng.standard_normal((p.n_paths, p.n_steps))

    prices = []
    for z, sigma, F0 in ((z_jkm, p.sigma_jkm, p.F0_jkm), (z_ttf, p.sigma_ttf, p.F0_ttf)):
        X = np.zeros((p.n_paths, p.n_steps))               # deviation from the "normal" level, per path and day
        x = np.zeros(p.n_paths)                            # starts at 0: today's price equals the forward
        for i in range(p.n_steps):
            x = a * x + sigma * step * z[:, i]             # "leash pull" + one random kick
            X[:, i] = x
        var_x = sigma**2 / (2 * p.kappa) * (1 - np.exp(-2 * p.kappa * t))   # Var(X_t), day by day
        prices.append(F0 * np.exp(X - 0.5 * var_x))        # -Var/2 keeps the average price equal to F0
    return prices[0], prices[1]


def pricing_average(S, window):
    """
    Average of the last `window` daily prices of each path [$/MMBtu].

    Cargo sale price and swap/put settlement are based on an average
    price over the pricing window (an "Asian" feature), not on a single day.
    """
    return S[:, -window:].mean(axis=1)


def pnl_cargo(A_jkm, p):
    """
    Unhedged cargo margin [$] = volume * (JKM average price - cost).
    What we earn (or lose) by selling the gas for more (or less) than it cost us.
    """
    return p.volume * (A_jkm - p.cost)


def pnl_swap(A_ttf, h, p):
    """
    P&L [$] of selling a TTF swap on h*volume at the fixed price K = TTF forward.

    The swap pays K - A_TTF. It costs nothing at inception because
    K equals the expected average price, so its expected P&L is zero.
    It cancels part of the cargo's ups and downs.
    """
    return h * p.volume * (p.F0_ttf - A_ttf)


def pnl_put(A_ttf, h, strike, premium, p):
    """
    P&L [$] of buying put options on h*volume of TTF, net of the premium paid.

    Asian put payoff = max(strike - A_TTF, 0) per MMBtu.
    We pay a premium up front; it pays out only if TTF
    falls below the strike, and we keep all the upside otherwise.
    """
    return h * p.volume * (np.maximum(strike - A_ttf, 0.0) - premium)


# ============================== 3. SOLVE ==============================
def min_variance_ratio(A_jkm, A_ttf):
    """
    Minimum-variance hedge ratio h* = Cov(JKM, TTF) / Var(TTF).

    The slope of the least-squares regression of JKM on TTF; it is the
    h that minimises the variance of the hedged margin.
    Hedging more adds risk.
    """
    return np.cov(A_jkm, A_ttf)[0, 1] / np.var(A_ttf, ddof=1)


def put_label(p):
    """Label of the put strategy, built from the strike ratio (e.g. 'Put TTF 90% (h*)')."""
    return f"Put TTF {p.put_strike_ratio:.0%} (h*)"


def run_strategies(p):
    """
    Simulate prices and compute the P&L of every strategy on the same random futures.

    All strategies share the same paths, so differences between them
    come only from the hedge, not from Monte Carlo noise. The put premium is the
    Monte Carlo fair value E[max(K - A_TTF, 0)] (no dealer margin, rates ~ 0).

    Returns
    -------
    results : dict {strategy name: ndarray of P&L [$] per path}
    info    : dict with h_star, premium, strike, price paths and pricing averages
    """
    S_jkm, S_ttf = simulate_prices(p)
    A_jkm = pricing_average(S_jkm, p.window)               # [$/MMBtu] cargo sale price, per path
    A_ttf = pricing_average(S_ttf, p.window)               # [$/MMBtu] hedge reference price, per path

    h_star = min_variance_ratio(A_jkm, A_ttf)              # [-] fraction of volume to hedge
    strike = p.put_strike_ratio * p.F0_ttf                 # [$/MMBtu]
    premium = np.maximum(strike - A_ttf, 0.0).mean()       # [$/MMBtu] fair put price

    base = pnl_cargo(A_jkm, p)                             # [$] unhedged margin
    results = {
        NO_HEDGE: base,
        SWAP_FULL: base + pnl_swap(A_ttf, 1.0, p),
        SWAP_MINVAR: base + pnl_swap(A_ttf, h_star, p),
        put_label(p): base + pnl_put(A_ttf, h_star, strike, premium, p),
    }
    info = dict(h_star=h_star, premium=premium, strike=strike, S_jkm=S_jkm, S_ttf=S_ttf,
                A_jkm=A_jkm, A_ttf=A_ttf)
    return results, info


def std_vs_hedge_ratio(A_jkm, A_ttf, p, h_grid):
    """
    Std of the hedged cargo P&L [$] for each swap hedge ratio h in `h_grid`.

    The minimum of this curve must coincide with h*, and it stays above zero 
    because JKM and TTF are not perfectly correlated (basis risk).
    """
    base = pnl_cargo(A_jkm, p)
    return np.array([np.std(base + pnl_swap(A_ttf, h, p)) for h in h_grid])


# ============================== 4. POST-PROCESS ==============================
def risk_table(results):
    """
    Print the risk metrics of each strategy (P&L in M$).

    Mean   : average P&L.
    Std    : spread of the outcomes (more = riskier).
    P5     : 5th percentile; in 95% of futures the P&L is at least this.
    ES5    : Expected Shortfall 5%, the average of the worst 5% (how bad it gets when it goes badly).
    P95    : 95th percentile; the upside.
    P(loss): fraction of futures with a negative P&L.
    Hedge eff.: share of the unhedged variance removed by the hedge.
    """
    base_var = results[NO_HEDGE].var()
    print(f"{'Strategy':<20}{'Mean':>8}{'Std':>8}{'P5':>8}{'ES5':>8}{'P95':>8}{'P(loss)':>9}{'Hedge eff.':>12}")
    for name, x in results.items():
        p5 = np.percentile(x, 5)
        es5 = x[x <= p5].mean()                            # average of the worst 5% outcomes
        eff = 1 - x.var() / base_var
        print(f"{name:<20}{x.mean()/1e6:8.2f}{x.std()/1e6:8.2f}{p5/1e6:8.2f}{es5/1e6:8.2f}"
              f"{np.percentile(x, 95)/1e6:8.2f}{(x < 0).mean():9.1%}{eff:12.1%}")


def verify_vs_analytical(p, info):
    """
    VERIFICATION: Monte Carlo vs closed-form price of a European put on TTF (Black-76).

    Test a European sibling (payoff on the last day only), where ln S_T is normal and
    Put = K*N(-d2) - F*N(-d1). If the simulator is right, the two prices agree
    within the Monte Carlo standard error.
    """
    K = p.put_strike_ratio * p.F0_ttf
    var_T = p.sigma_ttf**2 / (2 * p.kappa) * (1 - np.exp(-2 * p.kappa * p.T))   # Var(X_T)
    sd = np.sqrt(var_T)
    d1 = (np.log(p.F0_ttf / K) + 0.5 * var_T) / sd
    exact = K * norm_cdf(-(d1 - sd)) - p.F0_ttf * norm_cdf(-d1)                 # [$/MMBtu]

    payoff = np.maximum(K - info["S_ttf"][:, -1], 0.0)
    mc, se = payoff.mean(), payoff.std(ddof=1) / np.sqrt(p.n_paths)
    print(f"European put: MC {mc:.4f} +- {se:.4f} | closed form {exact:.4f} $/MMBtu")
    assert abs(mc - exact) < 4 * se, "Monte Carlo disagrees with the closed-form price"


def norm_cdf(x):
    """Standard normal cumulative distribution function N(x)."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def convergence_study(p, Ns=(1_000, 10_000, 100_000)):
    """
    CONVERGENCE: the Monte Carlo error must shrink as 1/sqrt(N).

    Technical: the standard error of the put premium is std(payoff)/sqrt(N), so
    'error * sqrt(N)' should stay roughly constant.
    To cut the error in half you need four times more simulated futures.
    """
    print(f"{'N':>8}{'put premium':>13}{'error':>10}{'error*sqrt(N)':>15}")
    for N in Ns:
        q = replace(p, n_paths=N)
        _, S_ttf = simulate_prices(q)
        payoff = np.maximum(q.put_strike_ratio * q.F0_ttf - pricing_average(S_ttf, q.window), 0.0)
        se = payoff.std(ddof=1) / np.sqrt(N)               # standard error of the mean
        print(f"{N:8d}{payoff.mean():13.4f}{se:10.4f}{se * np.sqrt(N):15.3f}")


def sensitivity_rho(p, rhos=(0.5, 0.7, 0.85, 0.95)):
    """
    SENSITIVITY: hedge effectiveness vs JKM-TTF correlation (expected ~ rho^2).

    The less TTF resembles JKM, the less a TTF hedge can protect us.
    """
    print(f"{'rho':>6}{'h*':>7}{'efficacy':>10}{'rho^2':>8}")
    for rho in rhos:
        res, info = run_strategies(replace(p, rho=rho, n_paths=30_000))
        eff = 1 - res[SWAP_MINVAR].var() / res[NO_HEDGE].var()
        print(f"{rho:6.2f}{info['h_star']:7.2f}{eff:10.1%}{rho**2:8.0%}")


def sanity_checks(p, info, h_grid, std_curve):
    """
    Physical and statistical sanity checks; the script stops with an error if one fails.

    1) Forward consistency: the simulated mean price must stay at the forward F0.
    2) h* must match its theoretical value rho*(sigma_J*F0_J)/(sigma_T*F0_T).
    3) The minimum of the std-vs-h curve must sit at h* (brute force vs formula).
    4) Limit case: identical JKM and TTF (rho = 1) -> perfect hedge, std -> 0.
    """
    # 1) Forward consistency
    for name, S, F0 in (("JKM", info["S_jkm"], p.F0_jkm), ("TTF", info["S_ttf"], p.F0_ttf)):
        err = S[:, -1].mean() / F0 - 1
        assert abs(err) < 0.005, f"{name} forward violated: {err:.3%}"
    # 2) Theoretical h* (volatility in $/MMBtu, not in %, hence the F0 factors)
    h_exp = p.rho * p.sigma_jkm * p.F0_jkm / (p.sigma_ttf * p.F0_ttf)
    assert abs(info["h_star"] - h_exp) < 0.05, "h* far from its theoretical value"
    print(f"h* = {info['h_star']:.2f} (theory ~ {h_exp:.2f}) | min-variance efficacy ~ rho^2 = {p.rho**2:.0%}")
    # 3) Brute-force minimum of the risk curve equals the regression slope h*
    h_min = h_grid[np.argmin(std_curve)]
    assert abs(h_min - info["h_star"]) <= 0.01, "std-vs-h minimum does not match h*"
    # 4) Limit case rho = 1
    q = replace(p, rho=1.0, sigma_ttf=p.sigma_jkm, F0_ttf=p.F0_jkm, n_paths=5000)
    res, _ = run_strategies(q)
    assert res[SWAP_MINVAR].std() / 1e6 < 1e-6, "rho = 1 limit not satisfied"
    print("Checks OK: forward, h* (formula and scan), rho = 1 limit (hedged std -> 0)")


def plot_price_and_pnl(results, info, p):
    """
    Figure 1. Left: 30 simulated JKM paths (the shaded zone is the pricing window).
    Right: P&L distribution of three strategies (dashed line = break-even).
    """
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    days = np.arange(1, p.n_steps + 1)
    for path in info["S_jkm"][:30]:
        ax[0].plot(days, path, lw=0.5, alpha=0.5)
    ax[0].plot(days, info["S_jkm"].mean(axis=0), "k", lw=2, label="mean of all paths")
    ax[0].axvspan(p.n_steps - p.window + 1, p.n_steps, color="grey", alpha=0.15, label="pricing window")
    ax[0].set(title="Simulated JKM price (30 paths)", xlabel="trading day", ylabel="$/MMBtu")
    ax[0].legend()
    for name in (NO_HEDGE, SWAP_MINVAR, put_label(p)):
        ax[1].hist(results[name] / 1e6, bins=80, alpha=0.5, label=name)
    ax[1].axvline(0, color="k", ls="--", lw=1)
    ax[1].set(title="Cargo P&L distribution", xlabel="P&L [M$]", ylabel="number of paths")
    ax[1].legend()
    fig.tight_layout()


def plot_hedge_curve(h_grid, std_curve, h_star):
    """
    Figure 2. Risk of the cargo (std of P&L) vs swap hedge ratio h.

    Reading guide: risk falls as we hedge, is lowest at h*, then rises again if we
    over-hedge. It never reaches zero: the remainder is the basis risk (JKM != TTF).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(h_grid, std_curve / 1e6, lw=2)
    ax.axhline(std_curve.min() / 1e6, color="grey", ls=":", lw=1)
    marks = ((0.0, "no hedge", (10, -5)), (1.0, "h = 1", (10, 12)), (h_star, f"h* = {h_star:.2f}", (-30, -38)))
    for h, label, offset in marks:
        y = np.interp(h, h_grid, std_curve) / 1e6
        ax.plot(h, y, "o", color="C3")
        ax.annotate(f"{label}: {y:.2f} M$", (h, y), textcoords="offset points", xytext=offset)
    ax.text(0.98, 0.04, "floor above zero = basis risk (JKM is not TTF)", transform=ax.transAxes,
            ha="right", color="grey")
    ax.set(title="Cargo risk vs hedge ratio (TTF swap)", xlabel="hedge ratio h (fraction of volume hedged)",
           ylabel="std of cargo P&L [M$]", ylim=(0, None))
    fig.tight_layout()


def main():
    p = Params()
    results, info = run_strategies(p)
    print(f"Put strike = {info['strike']:.2f} $/MMBtu | premium = {info['premium']:.2f} $/MMBtu "
          f"({info['premium'] * p.volume / 1e6:.2f} M$ for 100% of the volume)\n")
    risk_table(results)
    print()

    h_grid = np.arange(0.0, 1.5 + 1e-9, 0.01)              # [-] hedge ratios scanned for the risk curve
    std_curve = std_vs_hedge_ratio(info["A_jkm"], info["A_ttf"], p, h_grid)
    sanity_checks(p, info, h_grid, std_curve)

    print("\n-- Verification vs closed form --")
    verify_vs_analytical(p, info)
    print("\n-- Monte Carlo convergence --")
    convergence_study(p)
    print("\n-- Sensitivity to correlation --")
    sensitivity_rho(p)

    plot_price_and_pnl(results, info, p)
    plot_hedge_curve(h_grid, std_curve, info["h_star"])
    plt.show()                                             # opens the plot windows


if __name__ == "__main__":
    main()
