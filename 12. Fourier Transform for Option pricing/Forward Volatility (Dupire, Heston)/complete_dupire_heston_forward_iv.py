
#!/usr/bin/env python3
"""
Complete forward-start implied-volatility comparison:
Dupire local volatility vs Heston stochastic volatility.

No dividend yield q is used anywhere.

Flow implemented
----------------
1) Create / load a liquid grid of vanilla call option prices.
   - By default this script creates a synthetic but realistic call grid from a
     "true" Heston surface, so the example is fully standalone.
   - To use real data, replace the function create_liquid_call_grid().

2) Dupire flow:
   - Convert call prices into Black implied vols.
   - Fit a smooth implied-volatility surface to the liquid grid.
   - Use the smooth call-price surface in Dupire's formula to obtain local vol:
         sigma_loc^2(T,K) =
         [dC/dT + r K dC/dK] / [0.5 K^2 d2C/dK2]
   - Simulate local-vol Monte Carlo paths.
   - Price forward-starting options for multiple alpha values and T1/T2 pairs.
   - Invert the no-q Black forward-start formula to obtain forward IV.

3) Heston flow:
   - Calibrate the five Heston parameters to the same liquid call grid:
         v0, theta, kappa, xi, rho
   - Simulate Heston Monte Carlo paths.
   - Price the same forward-starting options.
   - Invert the no-q Black forward-start formula to obtain forward IV.

4) Compare plots:
   - Current vanilla smile for the same forward tenor
   - Dupire local-vol forward IV
   - Heston forward IV

Dependencies
------------
    pip install numpy scipy matplotlib

Run
---
    python complete_dupire_heston_forward_iv.py
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares
except Exception as exc:  # pragma: no cover
    least_squares = None


# =============================================================================
# 1) EDITABLE INPUTS
# =============================================================================

@dataclass(frozen=True)
class HestonParams:
    v0: float      # initial variance
    theta: float   # long-run variance
    kappa: float   # mean reversion
    xi: float      # vol of variance
    rho: float     # spot/variance correlation


@dataclass
class Inputs:
    # Basic market assumptions. No dividend yield q.
    S0: float = 100.0
    r: float = 0.0300

    # Liquid vanilla option grid used for calibration.
    # Strikes are created as K = strike_moneyness * F(0,T), where F(0,T)=S0 exp(rT).
    liquid_maturities: Tuple[float, ...] = (0.25, 0.50, 1.00, 2.00, 3.00)
    liquid_strike_moneyness: Tuple[float, ...] = (
        0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25
    )

    # Synthetic market generation.
    # These are used only to create a self-contained input call grid.
    # In real use, replace the call grid with market prices and ignore these.
    synthetic_market_heston: HestonParams = HestonParams(
        v0=0.0400,       # 20.00% spot variance vol
        theta=0.0400,    # 20.00% long-run vol
        kappa=1.8000,
        xi=0.5500,
        rho=-0.7000,
    )

    # Heston calibration controls.
    run_heston_calibration: bool = True
    heston_initial_guess: HestonParams = HestonParams(
        v0=0.0300,
        theta=0.0500,
        kappa=1.0000,
        xi=0.7000,
        rho=-0.4000,
    )
    heston_lower_bounds: HestonParams = HestonParams(
        v0=0.0001,
        theta=0.0001,
        kappa=0.0500,
        xi=0.0500,
        rho=-0.9500,
    )
    heston_upper_bounds: HestonParams = HestonParams(
        v0=1.0000,
        theta=1.0000,
        kappa=10.0000,
        xi=3.0000,
        rho=0.5000,
    )
    calibration_max_nfev: int = 80
    heston_integration_points: int = 300
    heston_integration_u_max: float = 100.0

    # Forward-start option definitions.
    # alpha is the strike multiplier at reset:
    #     K_T1 = alpha * S_T1
    #
    # The true forward log-moneyness is:
    #     k_fwd = ln(alpha) - r * (T2 - T1)
    forward_start_pairs: Tuple[Tuple[float, float], ...] = (
        (0.50, 1.00),
        (1.00, 2.00),
        (2.00, 3.00),
    )
    forward_alpha_grid: Tuple[float, ...] = (
        0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20
    )

    # Smooth IV surface fit used for Dupire.
    # Polynomial basis is in x = ln(K/F(0,T)) and T.
    min_fitted_iv: float = 0.0300
    max_fitted_iv: float = 1.5000

    # Dupire finite difference controls.
    dupire_dt: float = 1.0 / 365.0
    dupire_dk_rel: float = 0.0050
    min_local_vol: float = 0.0300
    max_local_vol: float = 2.0000

    # Local vol interpolation grid.
    local_vol_time_points: int = 90
    local_vol_spot_points: int = 181
    spot_grid_low_mult: float = 0.30
    spot_grid_high_mult: float = 2.30

    # Monte Carlo controls.
    num_paths: int = 50000
    steps_per_year: int = 120
    random_seed: int = 42
    use_antithetic: bool = True

    # Output controls.
    output_dir: str = "forward_iv_results"
    save_csv: bool = True
    save_plots: bool = True
    show_plots: bool = False


I = Inputs()


# =============================================================================
# 2) BASIC BLACK-SCHOLES / NORMAL UTILITIES
# =============================================================================

def norm_cdf(x):
    """
    Fast vectorized standard normal CDF using the Abramowitz-Stegun approximation.
    Good enough for pricing, calibration diagnostics, and IV inversion.
    """
    x = np.asarray(x, dtype=float)

    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    pdf = np.exp(-0.5 * ax * ax) / math.sqrt(2.0 * math.pi)
    poly = (((((b5 * t + b4) * t) + b3) * t + b2) * t + b1) * t
    cdf_pos = 1.0 - pdf * poly
    return np.where(x >= 0.0, cdf_pos, 1.0 - cdf_pos)


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_call_price(S0, K, r, T, vol):
    """
    No-q Black-Scholes call:
        C = S0 N(d1) - K exp(-rT) N(d2)
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    vol = np.asarray(vol, dtype=float)

    T_safe = np.maximum(T, 1.0e-10)
    vol_safe = np.maximum(vol, 1.0e-10)

    d1 = (np.log(S0 / K) + (r + 0.5 * vol_safe * vol_safe) * T_safe) / (
        vol_safe * np.sqrt(T_safe)
    )
    d2 = d1 - vol_safe * np.sqrt(T_safe)

    return S0 * norm_cdf(d1) - K * np.exp(-r * T_safe) * norm_cdf(d2)


def black_vega(S0, K, r, T, vol):
    T_safe = max(float(T), 1.0e-10)
    vol_safe = max(float(vol), 1.0e-10)
    d1 = (math.log(S0 / K) + (r + 0.5 * vol_safe * vol_safe) * T_safe) / (
        vol_safe * math.sqrt(T_safe)
    )
    return S0 * math.sqrt(T_safe) * float(norm_pdf(d1))


def implied_vol_call(price, S0, K, r, T, low=1.0e-8, high=5.0):
    """
    Bisection implied vol from no-q Black-Scholes call price.
    """
    intrinsic = max(S0 - K * math.exp(-r * T), 0.0)
    upper = S0

    if price <= intrinsic + 1.0e-12:
        return 0.0
    if price >= upper:
        return float("nan")

    def f(vol):
        return float(black_call_price(S0, K, r, T, vol) - price)

    f_low = f(low)
    f_high = f(high)

    while f_high < 0.0 and high < 20.0:
        high *= 2.0
        f_high = f(high)

    if f_low * f_high > 0.0:
        return float("nan")

    for _ in range(100):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < 1.0e-13:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def black_forward_start_call_price(S0, alpha, r, tau, vol):
    """
    No-q Black price of payoff:
        max(S_T2 - alpha * S_T1, 0)

    Because E[exp(-rT1) S_T1] = S0, the price is:
        C = S0 [N(d1) - alpha exp(-r tau) N(d2)]

    where:
        d1 = [-ln(alpha) + (r + 0.5 vol^2) tau] / [vol sqrt(tau)]
        d2 = d1 - vol sqrt(tau)
    """
    tau = max(float(tau), 1.0e-10)
    vol = max(float(vol), 1.0e-10)

    d1 = (-math.log(alpha) + (r + 0.5 * vol * vol) * tau) / (
        vol * math.sqrt(tau)
    )
    d2 = d1 - vol * math.sqrt(tau)

    return float(S0 * (norm_cdf(d1) - alpha * math.exp(-r * tau) * norm_cdf(d2)))


def implied_vol_forward_start(price, S0, alpha, r, tau, low=1.0e-8, high=5.0):
    """
    Bisection implied vol for the no-q Black forward-start formula.
    """
    intrinsic = max(S0 * (1.0 - alpha * math.exp(-r * tau)), 0.0)
    upper = S0

    if price <= intrinsic + 1.0e-12:
        return 0.0
    if price >= upper:
        return float("nan")

    def f(vol):
        return black_forward_start_call_price(S0, alpha, r, tau, vol) - price

    f_low = f(low)
    f_high = f(high)

    while f_high < 0.0 and high < 20.0:
        high *= 2.0
        f_high = f(high)

    if f_low * f_high > 0.0:
        return float("nan")

    for _ in range(100):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < 1.0e-13:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


# =============================================================================
# 3) HESTON PRICING AND CALIBRATION
# =============================================================================

def params_to_array(p: HestonParams):
    return np.array([p.v0, p.theta, p.kappa, p.xi, p.rho], dtype=float)


def array_to_params(x: Sequence[float]):
    return HestonParams(
        v0=float(x[0]),
        theta=float(x[1]),
        kappa=float(x[2]),
        xi=float(x[3]),
        rho=float(x[4]),
    )


def heston_cf(u, T, p: HestonParams, S0, r):
    """
    Characteristic function of log(S_T) under Heston.

    The implementation uses the common Gatheral-style representation and includes
    log(S0) + rT directly in the exponent. No dividend yield.
    """
    u = np.asarray(u, dtype=complex)

    i = 1j
    v0, theta, kappa, xi, rho = p.v0, p.theta, p.kappa, p.xi, p.rho

    d = np.sqrt((rho * xi * i * u - kappa) ** 2 + xi * xi * (i * u + u * u))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

    exp_neg_dt = np.exp(-d * T)

    C = (
        i * u * (math.log(S0) + r * T)
        + (kappa * theta / (xi * xi))
        * (
            (kappa - rho * xi * i * u - d) * T
            - 2.0 * np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
        )
    )

    D = ((kappa - rho * xi * i * u - d) / (xi * xi)) * (
        (1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt)
    )

    return np.exp(C + D * v0)


def heston_call_prices(S0, K, T, r, p: HestonParams, u_grid):
    """
    Heston call prices using P1/P2 Fourier probabilities.

        C = S0 P1 - K exp(-rT) P2

    Vectorized across strikes K for a single maturity T.
    """
    K = np.atleast_1d(K).astype(float)
    u = np.asarray(u_grid, dtype=float)

    cf_u = heston_cf(u, T, p, S0, r)
    cf_u_minus_i = heston_cf(u - 1j, T, p, S0, r)
    cf_minus_i = heston_cf(np.array([-1j]), T, p, S0, r)[0]

    log_k = np.log(K)

    expo = np.exp(-1j * u[:, None] * log_k[None, :])
    denom = 1j * u[:, None]

    integrand_p2 = np.real(expo * cf_u[:, None] / denom)
    integrand_p1 = np.real(expo * cf_u_minus_i[:, None] / (denom * cf_minus_i))

    p2 = 0.5 + np.trapezoid(integrand_p2, u, axis=0) / math.pi
    p1 = 0.5 + np.trapezoid(integrand_p1, u, axis=0) / math.pi

    prices = S0 * p1 - K * math.exp(-r * T) * p2

    # Conservative no-arbitrage clipping to avoid tiny numerical negatives.
    lower = np.maximum(S0 - K * math.exp(-r * T), 0.0)
    upper = S0 * np.ones_like(K)
    return np.clip(np.real(prices), lower, upper)


def heston_price_for_grid(options: List[Dict[str, float]], p: HestonParams, u_grid):
    """
    Price all options in the liquid grid under Heston.
    """
    prices_by_index = np.zeros(len(options), dtype=float)

    maturities = sorted(set(float(row["T"]) for row in options))
    for T in maturities:
        idx = [j for j, row in enumerate(options) if abs(row["T"] - T) < 1.0e-12]
        K = np.array([options[j]["K"] for j in idx], dtype=float)
        model_prices = heston_call_prices(I.S0, K, T, I.r, p, u_grid)
        prices_by_index[idx] = model_prices

    return prices_by_index


def calibrate_heston(options: List[Dict[str, float]], u_grid):
    """
    Calibrate the five Heston parameters to the liquid call grid.

    Objective is price error scaled approximately by one volatility point of vega.
    """
    if not I.run_heston_calibration:
        print("Heston calibration skipped. Using synthetic-market parameters.")
        return I.synthetic_market_heston

    if least_squares is None:
        print("SciPy was not found. Heston calibration skipped.")
        print("Install scipy or set run_heston_calibration=False.")
        return I.synthetic_market_heston

    market_prices = np.array([row["call_price"] for row in options], dtype=float)
    weights = []
    for row in options:
        vega_1volpt = black_vega(
            I.S0, row["K"], I.r, row["T"], max(row["implied_vol"], 0.01)
        ) * 0.01
        weights.append(max(vega_1volpt, 0.05))
    weights = np.array(weights, dtype=float)

    def objective(x):
        p = array_to_params(x)

        # Basic safety. Bounds should handle this, but these checks protect
        # against numerical issues during optimization.
        if p.v0 <= 0 or p.theta <= 0 or p.kappa <= 0 or p.xi <= 0 or abs(p.rho) >= 1:
            return 1.0e6 * np.ones_like(market_prices)

        model_prices = heston_price_for_grid(options, p, u_grid)
        return (model_prices - market_prices) / weights

    x0 = params_to_array(I.heston_initial_guess)
    lb = params_to_array(I.heston_lower_bounds)
    ub = params_to_array(I.heston_upper_bounds)

    print("\nCalibrating Heston parameters to the liquid call grid...")
    result = least_squares(
        objective,
        x0=x0,
        bounds=(lb, ub),
        max_nfev=I.calibration_max_nfev,
        xtol=1.0e-7,
        ftol=1.0e-7,
        gtol=1.0e-7,
        verbose=0,
    )

    p = array_to_params(result.x)

    print("Heston calibration status:", result.message)
    print(f"Function evaluations     : {result.nfev}")
    print(f"Weighted RMSE            : {math.sqrt(2.0 * result.cost / len(market_prices)):.6f}")
    print("Calibrated Heston params :")
    print(f"    v0    = {p.v0:.6f}  -> sqrt(v0)    = {math.sqrt(p.v0):.2%}")
    print(f"    theta = {p.theta:.6f}  -> sqrt(theta) = {math.sqrt(p.theta):.2%}")
    print(f"    kappa = {p.kappa:.6f}")
    print(f"    xi    = {p.xi:.6f}")
    print(f"    rho   = {p.rho:.6f}")

    return p


# =============================================================================
# 4) LIQUID CALL GRID INPUT
# =============================================================================

def make_heston_integration_grid():
    return np.linspace(
        1.0e-5,
        I.heston_integration_u_max,
        I.heston_integration_points,
        dtype=float,
    )


def create_liquid_call_grid(u_grid):
    """
    Creates a synthetic liquid call grid.

    In production, replace this function with your market loader, returning rows:
        {"T": T, "K": K, "call_price": price}

    This default version creates prices from I.synthetic_market_heston and then
    calculates Black implied vols. That gives us a realistic no-arbitrage market
    surface and lets us test whether the Heston calibration can recover the
    original parameters.
    """
    rows: List[Dict[str, float]] = []

    for T in I.liquid_maturities:
        F = I.S0 * math.exp(I.r * T)
        strikes = np.array(I.liquid_strike_moneyness, dtype=float) * F
        prices = heston_call_prices(
            I.S0, strikes, T, I.r, I.synthetic_market_heston, u_grid
        )

        for mny, K, price in zip(I.liquid_strike_moneyness, strikes, prices):
            iv = implied_vol_call(float(price), I.S0, float(K), I.r, float(T))
            rows.append(
                {
                    "T": float(T),
                    "strike_moneyness": float(mny),
                    "K": float(K),
                    "call_price": float(price),
                    "implied_vol": float(iv),
                }
            )

    return rows


# =============================================================================
# 5) SMOOTH IV SURFACE FIT FOR DUPIRE
# =============================================================================

def iv_surface_features(T, K):
    """
    Polynomial features for fitting implied volatility.

    x = log-forward moneyness = ln(K / F(0,T))

    T and K may be scalars, arrays, or one scalar plus one array.
    This function broadcasts them to the same shape and returns a 2D design matrix.
    """
    T, K = np.broadcast_arrays(np.asarray(T, dtype=float), np.asarray(K, dtype=float))
    T = T.ravel()
    K = K.ravel()

    F = I.S0 * np.exp(I.r * T)
    x = np.log(np.maximum(K, 1.0e-12) / F)
    sqrtT = np.sqrt(np.maximum(T, 1.0e-8))
    logT = np.log1p(np.maximum(T, 0.0))

    return np.column_stack(
        [
            np.ones_like(x),
            x,
            x * x,
            x * x * x,
            sqrtT,
            logT,
            x * sqrtT,
            x * x * sqrtT,
            x * logT,
            x * x * logT,
        ]
    )


def fit_smooth_iv_surface(options: List[Dict[str, float]]):
    """
    Least-squares polynomial fit to the input implied vols.
    """
    T = np.array([row["T"] for row in options], dtype=float)
    K = np.array([row["K"] for row in options], dtype=float)
    y = np.array([row["implied_vol"] for row in options], dtype=float)

    X = iv_surface_features(T, K)
    beta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)

    fit = X @ beta
    rmse = math.sqrt(float(np.mean((fit - y) ** 2)))

    print("\nSmooth IV surface fitted for Dupire.")
    print(f"Number of grid options : {len(options)}")
    print(f"IV fit RMSE            : {rmse:.6%}")

    return beta


def fitted_implied_vol(T, K, beta):
    T_arr, K_arr = np.broadcast_arrays(np.asarray(T, dtype=float), np.asarray(K, dtype=float))
    out_shape = T_arr.shape
    X = iv_surface_features(T_arr, K_arr)
    vol = (X @ beta).reshape(out_shape)
    return np.clip(vol, I.min_fitted_iv, I.max_fitted_iv)


def fitted_call_price(T, K, beta):
    vol = fitted_implied_vol(T, K, beta)
    return black_call_price(I.S0, K, I.r, T, vol)


# =============================================================================
# 6) DUPIRE LOCAL VOLATILITY FROM FITTED CALL SURFACE
# =============================================================================

def dupire_local_vol(T, K, beta):
    """
    Dupire formula with no dividend yield:

        sigma_loc^2(T,K) =
        [dC/dT + r K dC/dK] / [0.5 K^2 d2C/dK2]
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    T0 = np.maximum(T, 2.0 * I.dupire_dt)
    dT = I.dupire_dt
    T_up = T0 + dT
    T_dn = np.maximum(T0 - dT, 1.0e-6)

    dK = np.maximum(I.dupire_dk_rel * K, 0.01 * I.S0)
    K_up = K + dK
    K_dn = np.maximum(K - dK, 1.0e-8)

    C_T_up = fitted_call_price(T_up, K, beta)
    C_T_dn = fitted_call_price(T_dn, K, beta)
    dC_dT = (C_T_up - C_T_dn) / (T_up - T_dn)

    C_K_up = fitted_call_price(T0, K_up, beta)
    C_K_mid = fitted_call_price(T0, K, beta)
    C_K_dn = fitted_call_price(T0, K_dn, beta)

    dC_dK = (C_K_up - C_K_dn) / (K_up - K_dn)
    d2C_dK2 = (C_K_up - 2.0 * C_K_mid + C_K_dn) / (dK * dK)

    numerator = dC_dT + I.r * K * dC_dK
    denominator = 0.5 * K * K * d2C_dK2

    lv2 = np.full_like(K, np.nan, dtype=float)
    valid = denominator > 1.0e-12
    lv2[valid] = numerator[valid] / denominator[valid]

    atm_iv = fitted_implied_vol(np.maximum(T0, 1.0e-6), I.S0 * np.exp(I.r * T0), beta)
    lv2 = np.where(np.isfinite(lv2), lv2, atm_iv * atm_iv)

    lv2 = np.clip(lv2, I.min_local_vol ** 2, I.max_local_vol ** 2)
    return np.sqrt(lv2)


def build_local_vol_grid(beta):
    max_T2 = max(pair[1] for pair in I.forward_start_pairs)

    t_grid = np.linspace(1.0e-4, max_T2, I.local_vol_time_points)
    s_grid = np.linspace(
        I.spot_grid_low_mult * I.S0,
        I.spot_grid_high_mult * I.S0,
        I.local_vol_spot_points,
    )

    lv_grid = np.zeros((len(t_grid), len(s_grid)), dtype=float)
    for i, t in enumerate(t_grid):
        lv_grid[i, :] = dupire_local_vol(t, s_grid, beta)

    return t_grid, s_grid, lv_grid


def interpolate_local_vol(t, S, t_grid, s_grid, lv_grid):
    S = np.asarray(S, dtype=float)
    S_clip = np.clip(S, s_grid[0], s_grid[-1])
    t_clip = float(np.clip(t, t_grid[0], t_grid[-1]))

    i = int(np.searchsorted(t_grid, t_clip) - 1)
    i = int(np.clip(i, 0, len(t_grid) - 2))

    t0 = t_grid[i]
    t1 = t_grid[i + 1]
    w = (t_clip - t0) / (t1 - t0)

    vol0 = np.interp(S_clip, s_grid, lv_grid[i, :])
    vol1 = np.interp(S_clip, s_grid, lv_grid[i + 1, :])

    return (1.0 - w) * vol0 + w * vol1


# =============================================================================
# 7) MONTE CARLO SIMULATION
# =============================================================================

def required_snapshot_times():
    times = sorted(set([0.0] + [t for pair in I.forward_start_pairs for t in pair]))
    return times


def make_mc_time_grid():
    max_T2 = max(pair[1] for pair in I.forward_start_pairs)
    dt = 1.0 / I.steps_per_year
    n_steps = int(round(max_T2 / dt))
    n_steps = max(n_steps, 1)
    # Redefine dt so the final time lands exactly on max_T2.
    dt = max_T2 / n_steps
    times = np.linspace(0.0, max_T2, n_steps + 1)
    return times, dt


def nearest_step_indices(times, snapshot_times):
    idx = {}
    for t in snapshot_times:
        j = int(np.argmin(np.abs(times - t)))
        idx[float(t)] = j
    return idx


def prepare_random_normals(n_steps, n_paths, seed, dimensions):
    """
    Returns normal random numbers.

    dimensions=1:
        shape (n_steps, n_paths)

    dimensions=2:
        shape (n_steps, n_paths, 2)
    """
    rng = np.random.default_rng(seed)

    if I.use_antithetic:
        n_paths = 2 * (n_paths // 2)
        half = n_paths // 2

        if dimensions == 1:
            z_half = rng.standard_normal((n_steps, half))
            return np.concatenate([z_half, -z_half], axis=1)

        z_half = rng.standard_normal((n_steps, half, dimensions))
        return np.concatenate([z_half, -z_half], axis=1)

    if dimensions == 1:
        return rng.standard_normal((n_steps, n_paths))

    return rng.standard_normal((n_steps, n_paths, dimensions))


def simulate_dupire_paths(beta):
    """
    Simulate the Dupire local-vol model:
        dS/S = r dt + sigma_loc(t,S) dW

    Returns a dictionary:
        snapshots[t] = array of S_t paths
    """
    print("\nBuilding Dupire local-vol grid...")
    t_grid, s_grid, lv_grid = build_local_vol_grid(beta)

    times, dt = make_mc_time_grid()
    n_steps = len(times) - 1
    n_paths = 2 * (I.num_paths // 2) if I.use_antithetic else I.num_paths

    snapshot_times = required_snapshot_times()
    snapshot_indices = nearest_step_indices(times, snapshot_times)
    reverse_snapshot_indices = {v: k for k, v in snapshot_indices.items()}

    print("Simulating Dupire local-vol paths...")
    Z = prepare_random_normals(n_steps, n_paths, I.random_seed, dimensions=1)

    S = np.full(n_paths, I.S0, dtype=float)
    snapshots: Dict[float, np.ndarray] = {0.0: S.copy()}

    for step in range(n_steps):
        t = times[step]
        vol = interpolate_local_vol(t, S, t_grid, s_grid, lv_grid)
        S *= np.exp((I.r - 0.5 * vol * vol) * dt + vol * math.sqrt(dt) * Z[step])

        if step + 1 in reverse_snapshot_indices:
            snapshots[reverse_snapshot_indices[step + 1]] = S.copy()

    return snapshots


def simulate_heston_paths(p: HestonParams):
    """
    Simulate the Heston model using full truncation Euler:

        dS/S = r dt + sqrt(v) dW_S
        dv   = kappa(theta-v)dt + xi sqrt(v)dW_v

    with corr(dW_S,dW_v)=rho.
    """
    times, dt = make_mc_time_grid()
    n_steps = len(times) - 1
    n_paths = 2 * (I.num_paths // 2) if I.use_antithetic else I.num_paths

    snapshot_times = required_snapshot_times()
    snapshot_indices = nearest_step_indices(times, snapshot_times)
    reverse_snapshot_indices = {v: k for k, v in snapshot_indices.items()}

    print("\nSimulating Heston paths...")
    Z = prepare_random_normals(n_steps, n_paths, I.random_seed + 17, dimensions=2)

    S = np.full(n_paths, I.S0, dtype=float)
    v = np.full(n_paths, p.v0, dtype=float)

    snapshots: Dict[float, np.ndarray] = {0.0: S.copy()}

    rho = p.rho
    sqrt_one_minus_rho2 = math.sqrt(max(1.0 - rho * rho, 0.0))

    for step in range(n_steps):
        z_v = Z[step, :, 0]
        z_ind = Z[step, :, 1]
        z_s = rho * z_v + sqrt_one_minus_rho2 * z_ind

        v_pos = np.maximum(v, 0.0)

        S *= np.exp((I.r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s)

        v = (
            v
            + p.kappa * (p.theta - v_pos) * dt
            + p.xi * np.sqrt(v_pos * dt) * z_v
        )
        v = np.maximum(v, 0.0)

        if step + 1 in reverse_snapshot_indices:
            snapshots[reverse_snapshot_indices[step + 1]] = S.copy()

    return snapshots


# =============================================================================
# 8) FORWARD-START PRICING AND IV INVERSION
# =============================================================================

def price_forward_start_from_snapshots(snapshots, T1, T2, alpha):
    S1 = snapshots[float(T1)]
    S2 = snapshots[float(T2)]
    payoff = np.maximum(S2 - alpha * S1, 0.0)
    return math.exp(-I.r * T2) * float(np.mean(payoff))


def current_vanilla_iv_same_forward_tenor(alpha, tau, beta):
    """
    Reference current vanilla IV for maturity tau and same alpha convention.

    We use K = alpha * F(0,tau), so this is the current vanilla smile as a
    function of the same strike/forward ratio over the forward tenor tau.
    """
    F = I.S0 * math.exp(I.r * tau)
    K = alpha * F
    return float(fitted_implied_vol(tau, K, beta))


def compute_forward_iv_results(dupire_snapshots, heston_snapshots, beta):
    rows = []

    for T1, T2 in I.forward_start_pairs:
        tau = T2 - T1

        for alpha in I.forward_alpha_grid:
            lv_price = price_forward_start_from_snapshots(dupire_snapshots, T1, T2, alpha)
            heston_price = price_forward_start_from_snapshots(heston_snapshots, T1, T2, alpha)

            lv_iv = implied_vol_forward_start(lv_price, I.S0, alpha, I.r, tau)
            heston_iv = implied_vol_forward_start(heston_price, I.S0, alpha, I.r, tau)

            current_iv = current_vanilla_iv_same_forward_tenor(alpha, tau, beta)
            k_fwd = math.log(alpha) - I.r * tau

            rows.append(
                {
                    "T1": float(T1),
                    "T2": float(T2),
                    "tau": float(tau),
                    "alpha": float(alpha),
                    "k_fwd": float(k_fwd),
                    "current_vanilla_iv_same_tenor": float(current_iv),
                    "dupire_forward_price": float(lv_price),
                    "dupire_forward_iv": float(lv_iv),
                    "heston_forward_price": float(heston_price),
                    "heston_forward_iv": float(heston_iv),
                }
            )

    return rows


# =============================================================================
# 9) REPORTING
# =============================================================================

def ensure_output_dir():
    os.makedirs(I.output_dir, exist_ok=True)


def save_liquid_grid(options):
    if not I.save_csv:
        return
    ensure_output_dir()
    path = os.path.join(I.output_dir, "liquid_call_grid.csv")
    fields = ["T", "strike_moneyness", "K", "call_price", "implied_vol"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in options:
            writer.writerow({k: row[k] for k in fields})
    print(f"Saved liquid call grid: {path}")


def save_results(rows):
    if not I.save_csv:
        return
    ensure_output_dir()
    path = os.path.join(I.output_dir, "forward_iv_comparison.csv")
    fields = [
        "T1",
        "T2",
        "tau",
        "alpha",
        "k_fwd",
        "current_vanilla_iv_same_tenor",
        "dupire_forward_price",
        "dupire_forward_iv",
        "heston_forward_price",
        "heston_forward_iv",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print(f"Saved forward IV results: {path}")


def print_results(rows):
    print("\nForward-start implied-volatility comparison")
    print("=" * 110)
    print(
        f"{'T1':>5} {'T2':>5} {'alpha':>8} {'k_fwd':>10} "
        f"{'current IV':>12} {'Dupire IV':>12} {'Heston IV':>12} "
        f"{'Dupire px':>12} {'Heston px':>12}"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"{row['T1']:5.2f} {row['T2']:5.2f} {row['alpha']:8.2f} "
            f"{row['k_fwd']:10.4f} "
            f"{row['current_vanilla_iv_same_tenor']:12.4%} "
            f"{row['dupire_forward_iv']:12.4%} "
            f"{row['heston_forward_iv']:12.4%} "
            f"{row['dupire_forward_price']:12.6f} "
            f"{row['heston_forward_price']:12.6f}"
        )

    print("-" * 110)

    for T1, T2 in I.forward_start_pairs:
        subset = [row for row in rows if abs(row["T1"] - T1) < 1e-12 and abs(row["T2"] - T2) < 1e-12]
        current = np.array([row["current_vanilla_iv_same_tenor"] for row in subset])
        lv = np.array([row["dupire_forward_iv"] for row in subset])
        hs = np.array([row["heston_forward_iv"] for row in subset])

        lv_ratio = np.nanstd(lv) / np.nanstd(current)
        hs_ratio = np.nanstd(hs) / np.nanstd(current)

        print(
            f"T1={T1:.2f}, T2={T2:.2f}: "
            f"std(Dupire fwd IV)/std(current IV) = {lv_ratio:.2%}, "
            f"std(Heston fwd IV)/std(current IV) = {hs_ratio:.2%}"
        )


def plot_results(rows):
    if not I.save_plots and not I.show_plots:
        return

    ensure_output_dir()

    for T1, T2 in I.forward_start_pairs:
        subset = [row for row in rows if abs(row["T1"] - T1) < 1e-12 and abs(row["T2"] - T2) < 1e-12]
        subset = sorted(subset, key=lambda x: x["alpha"])

        alpha = np.array([row["alpha"] for row in subset])
        current = 100.0 * np.array([row["current_vanilla_iv_same_tenor"] for row in subset])
        dupire = 100.0 * np.array([row["dupire_forward_iv"] for row in subset])
        heston = 100.0 * np.array([row["heston_forward_iv"] for row in subset])

        plt.figure(figsize=(9.5, 5.5))
        plt.plot(alpha, current, marker="o", linestyle="--", label="Current vanilla IV, same tenor")
        plt.plot(alpha, dupire, marker="s", label="Dupire local-vol forward IV")
        plt.plot(alpha, heston, marker="^", label="Heston forward IV")

        plt.xlabel(r"Forward-start strike multiplier $\alpha = K_{T_1}/S_{T_1}$")
        plt.ylabel("Black implied volatility (%)")
        plt.title(f"Forward-start IV comparison: T1={T1:.2f}, T2={T2:.2f}")
        plt.grid(True, alpha=0.30)
        plt.legend()
        plt.tight_layout()

        if I.save_plots:
            path = os.path.join(I.output_dir, f"forward_iv_T1_{T1:.2f}_T2_{T2:.2f}.png")
            plt.savefig(path, dpi=160)
            print(f"Saved plot: {path}")

        if I.show_plots:
            plt.show()

        plt.close()


# =============================================================================
# 10) MAIN
# =============================================================================

def main():
    print("Complete Dupire vs Heston forward-start IV implementation")
    print("No dividend yield q")
    print("=" * 80)
    print(f"S0                    : {I.S0:.4f}")
    print(f"r                     : {I.r:.4%}")
    print(f"Liquid maturities     : {I.liquid_maturities}")
    print(f"Liquid moneyness grid : {I.liquid_strike_moneyness}")
    print(f"Forward pairs         : {I.forward_start_pairs}")
    print(f"Forward alpha grid    : {I.forward_alpha_grid}")
    print(f"MC paths              : {I.num_paths:,}")
    print(f"Steps per year        : {I.steps_per_year}")

    u_grid = make_heston_integration_grid()

    print("\nCreating assumed liquid call price grid...")
    options = create_liquid_call_grid(u_grid)
    save_liquid_grid(options)

    beta = fit_smooth_iv_surface(options)

    heston_params = calibrate_heston(options, u_grid)

    dupire_snapshots = simulate_dupire_paths(beta)
    heston_snapshots = simulate_heston_paths(heston_params)

    rows = compute_forward_iv_results(dupire_snapshots, heston_snapshots, beta)

    print_results(rows)
    save_results(rows)
    plot_results(rows)

    print("\nDone.")
    print(f"Output folder: {I.output_dir}")
    print("\nNotes:")
    print("  alpha is K_T1 / S_T1.")
    print("  true forward log-moneyness is k_fwd = ln(alpha) - r * (T2 - T1).")
    print("  To use real market data, replace create_liquid_call_grid().")


if __name__ == "__main__":
    main()
