# LNG Cargo Hedging with Monte Carlo

How risky is the margin of an LNG cargo between purchase and sale, and how much does a hedge with derivatives reduce that risk?
A small, self-contained Python study: Monte Carlo price simulation, three hedging strategies, risk metrics, and a validation suite.

## Problem

A trader buys one cargo (3.5 M MMBtu) at a fixed all-in cost of 9.5 $/MMBtu (e.g. an Henry Hub-linked contract plus liquefaction and shipping) and sells it in 3 months at the **JKM** price (average of the last 21 trading days). The margin is

`P&L = volume x (average JKM - cost)`

and the average JKM price is uncertain. The risk is hedged with derivatives on **TTF**, a more liquid benchmark that is *not* the same price as JKM, so some **basis risk** remains.

## Model

Prices follow a one-factor mean-reverting model (Schwartz, 1997), consistent with the forward curve (`E[S_t] = F0`). JKM and TTF are correlated (rho = 0.85). Each simulation uses 100,000 paths of 63 daily steps (exact discretisation).

| Element | Technical meaning | Plain-language intuition |
|---|---|---|
| `dX = -kappa X dt + sigma dW` | Ornstein-Uhlenbeck (mean reversion) | A dog on an elastic leash: it wanders randomly but is pulled back to its owner (the forward price) |
| `S = F0 exp(X - Var/2)` | Price, always positive, mean = F0 | The `-Var/2` term corrects for the fact that +50% then -50% does not bring you back to 100 |
| Monte Carlo | N random futures, statistics on the P&L | Replay thousands of possible futures and count what happens; error scales as 1/sqrt(N) |
| Swap TTF | Pays `K - A_TTF` | A deal that pays when the price falls: cancels part of the ups and downs |
| `h* = Cov(JKM,TTF)/Var(TTF)` | Minimum-variance hedge ratio (least-squares slope) | How much of the volume to hedge to remove the most risk |
| Put TTF | Pays `max(strike - A_TTF, 0)` | Insurance: pay a premium, keep the upside |
| P5, ES5 | 5th percentile, mean of the worst 5% | "In 95% of futures I do at least P5" / "how bad it gets when it goes badly" |

## Strategies compared

1. **No hedge**
2. **Swap TTF h=1**: hedge 100% of the volume at the TTF forward
3. **Swap TTF h\***: hedge the minimum-variance share of the volume
4. **Put TTF 90% (h\*)**: Asian put on TTF with strike at 90% of the forward, fair premium from the simulation

## Validation

- **Forward consistency**: the simulated mean price stays at the forward.
- **Verification**: Monte Carlo European put vs closed form (Black-76): 0.609 +- 0.003 vs 0.612 $/MMBtu.
- **Convergence**: error x sqrt(N) is constant, so the error shrinks as 1/sqrt(N).
- **Limit case**: rho = 1 (identical indices) gives a perfect hedge, std -> 0.
- **Minimum check**: the brute-force minimum of the risk-vs-h curve matches the analytical h*.
- **Sensitivity**: hedge efficacy is approximately rho^2 for rho between 0.5 and 0.95.

## Results (P&L in M$, 100,000 paths)

| Strategy | Mean | Std | P5 | ES5 | P95 | P(loss) |
|---|---|---|---|---|---|---|
| No hedge | 8.79 | 9.17 | -4.43 | -6.87 | 25.38 | 16.3% |
| Swap TTF h=1 | 8.79 | 5.05 | 0.55 | -1.97 | 17.09 | 4.1% |
| Swap TTF h* (0.85) | 8.79 | 4.87 | 1.33 | -0.67 | 17.22 | 2.7% |
| Put TTF 90% (h*) | 8.79 | 7.80 | -1.03 | -2.61 | 23.84 | 8.3% |

The swap roughly halves the risk but also caps the upside. The put protects against the downside and keeps more upside. The mean is identical because both hedges are priced at fair value. Hedging 85% (h\*) beats hedging 100%: over-hedging adds risk.

## Plots

Running the script opens two windows (nothing is saved to disk):

1. **Simulated JKM paths** (with the pricing window) and the **P&L distribution** of three strategies.
2. **Cargo risk vs hedge ratio**: standard deviation of the P&L as a function of h. The minimum sits at h\*, and the floor above zero is the basis risk.

## Usage

```bash
pip install -r requirements.txt
python lng_hedge.py      # prints tables and checks, then opens the plots
```

Python 3.8+. All inputs are in the `Params` dataclass at the top of `lng_hedge.py`.

## Code structure

`lng_hedge.py` is organised in four blocks: **1. Input** (`Params`), **2. Model** (price simulation, P&L of cargo, swap and put), **3. Solve** (hedge ratio, strategy comparison), **4. Post-process** (risk table, validation, plots).

## Limitations

- Market parameters are **illustrative**: calibrate them on historical JKM/TTF data before real use.
- The put premium is a fair value (no dealer margin, rates ~ 0 over 3 months), which is why the mean P&L is the same for all strategies.
- One-factor model: no seasonality and no price jumps (relevant for gas in winter).
- Margin calls, transaction costs and volume risk are not modelled.

## Reference

E. Schwartz (1997), *The Stochastic Behavior of Commodity Prices: Implications for Valuation and Hedging*, Journal of Finance 52(3).
F. Black (1976), *The Pricing of Commodity Contracts*, Journal of Financial Economics 3.
