"""
Heston model teaching script
----------------------------
Standalone Python implementation for:
1. Heston characteristic function
2. Fourier inverse option pricing
3. Carr-Madan FFT option pricing for a range of strikes
4. Black-Scholes implied volatility inversion
5. Heston implied volatility surface plots
6. End-to-end calibration to a representative liquid option quote grid
7. Finite-difference Greeks under Heston

Dependencies: numpy, scipy, matplotlib
No QuantLib is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize
from scipy.stats import norm


@dataclass(frozen=True)
class HestonParams:
    """Container for Heston variance process parameters."""
    v0: float       # initial variance
    kappa: float    # variance mean-reversion speed
    theta: float    # long-run variance
    xi: float       # volatility of variance (vol-of-vol)
    rho: float      # correlation between spot and variance Brownian motions


def black_scholes_call(S0: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """Black-Scholes price of a European call option."""
    if T <= 0:
        return max(S0 - K, 0.0)
    if vol <= 0:
        return max(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    sigma_sqrt_t = vol * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """Black-Scholes price of a European put option."""
    call = black_scholes_call(S0, K, T, r, q, vol)
    return call - S0 * math.exp(-q * T) + K * math.exp(-r * T)


def bs_vega(S0: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """Black-Scholes vega per 1.00 volatility point."""
    if T <= 0 or vol <= 0:
        return 0.0
    sigma_sqrt_t = vol * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / sigma_sqrt_t
    return S0 * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def implied_vol_from_call(price: float, S0: float, K: float, T: float, r: float, q: float) -> float:
    """Black-Scholes implied volatility from a call price."""
    intrinsic = max(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    upper_bound = S0 * math.exp(-q * T)
    price = min(max(price, intrinsic + 1.0e-12), upper_bound - 1.0e-12)

    def objective(vol: float) -> float:
        return black_scholes_call(S0, K, T, r, q, vol) - price

    try:
        return brentq(objective, 1.0e-6, 5.0, maxiter=200)
    except ValueError:
        return np.nan


def heston_char_func(u: complex, S0: float, T: float, r: float, q: float, p: HestonParams) -> complex:
    """
    Heston characteristic function for X_T = log(S_T).

    This uses the common 'little Heston trap' parameterization:
        E[exp(i u log S_T)] = exp(C(u,T) + D(u,T) v0 + i u log(S0))
    """
    i = 1j
    a = p.kappa * p.theta
    b = p.kappa
    sigma = p.xi
    rho = p.rho

    d = np.sqrt((rho * sigma * i * u - b) ** 2 + sigma ** 2 * (i * u + u ** 2))
    g = (b - rho * sigma * i * u - d) / (b - rho * sigma * i * u + d)

    exp_neg_dT = np.exp(-d * T)
    C = (r - q) * i * u * T + (a / sigma ** 2) * (
        (b - rho * sigma * i * u - d) * T
        - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
    )
    D = ((b - rho * sigma * i * u - d) / sigma ** 2) * (
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    )
    return np.exp(C + D * p.v0 + i * u * math.log(S0))


def heston_call_fourier_inverse(S0: float, K: float, T: float, r: float, q: float, p: HestonParams) -> float:
    """European call price using the Heston P1/P2 Fourier inversion formula."""
    logK = math.log(K)
    phi_minus_i = heston_char_func(-1j, S0, T, r, q, p)

    def integrand_p1(u: float) -> float:
        z = np.exp(-1j * u * logK) * heston_char_func(u - 1j, S0, T, r, q, p) / (1j * u * phi_minus_i)
        return np.real(z)

    def integrand_p2(u: float) -> float:
        z = np.exp(-1j * u * logK) * heston_char_func(u, S0, T, r, q, p) / (1j * u)
        return np.real(z)

    p1 = 0.5 + quad(integrand_p1, 1.0e-8, 150.0, limit=500, epsabs=1.0e-8, epsrel=1.0e-8)[0] / math.pi
    p2 = 0.5 + quad(integrand_p2, 1.0e-8, 150.0, limit=500, epsabs=1.0e-8, epsrel=1.0e-8)[0] / math.pi
    return S0 * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2


def heston_fft_call_grid(
    S0: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carr-Madan FFT call prices on an equally spaced log-strike grid.

    Returns
    -------
    strikes : np.ndarray
        Strike grid exp(k_j).
    call_prices : np.ndarray
        Corresponding European call prices.
    """
    i = 1j
    lambd = 2.0 * math.pi / (N * eta)
    b = 0.5 * N * lambd
    v = eta * np.arange(N)
    k = -b + lambd * np.arange(N)

    shifted_u = v - (alpha + 1.0) * i
    denominator = alpha ** 2 + alpha - v ** 2 + i * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * T) * heston_char_func(shifted_u, S0, T, r, q, p) / denominator

    # Simpson weights improve convergence of the trapezoidal FFT approximation.
    weights = np.ones(N)
    weights[0] = 1.0 / 3.0
    weights[1::2] = 4.0 / 3.0
    weights[2::2] = 2.0 / 3.0

    fft_input = np.exp(i * b * v) * psi * eta * weights
    fft_values = np.fft.fft(fft_input)
    call_prices = np.exp(-alpha * k) * np.real(fft_values) / math.pi
    strikes = np.exp(k)

    valid = (strikes > 1.0e-6) & np.isfinite(call_prices) & (call_prices > 0.0)
    return strikes[valid], call_prices[valid]


def heston_call_fft(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> float:
    """European call price at one strike by interpolating the Carr-Madan FFT grid."""
    strikes, prices = heston_fft_call_grid(S0, T, r, q, p, alpha=alpha, N=N, eta=eta)
    interpolator = interp1d(strikes, prices, kind="cubic", fill_value="extrapolate", assume_sorted=True)
    return float(interpolator(K))


def heston_put_fft(S0: float, K: float, T: float, r: float, q: float, p: HestonParams) -> float:
    """European put price from put-call parity."""
    call = heston_call_fft(S0, K, T, r, q, p)
    return call - S0 * math.exp(-q * T) + K * math.exp(-r * T)


def call_prices_for_strikes(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    N: int = 4096,
    eta: float = 0.25,
) -> np.ndarray:
    """Vector of FFT-interpolated call prices for a fixed maturity and multiple strikes."""
    grid_strikes, grid_prices = heston_fft_call_grid(S0, T, r, q, p, N=N, eta=eta)
    interpolator = interp1d(grid_strikes, grid_prices, kind="cubic", fill_value="extrapolate", assume_sorted=True)
    return np.maximum(interpolator(strikes), 0.0)


def implied_vol_surface(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    q: float,
    p: HestonParams,
    N: int = 4096,
    eta: float = 0.25,
) -> np.ndarray:
    """Heston implied volatility surface on a strike/maturity grid."""
    iv = np.empty((len(maturities), len(strikes)))
    for ti, T in enumerate(maturities):
        prices = call_prices_for_strikes(S0, strikes, T, r, q, p, N=N, eta=eta)
        for ki, K in enumerate(strikes):
            iv[ti, ki] = implied_vol_from_call(float(prices[ki]), S0, float(K), float(T), r, q)
    return iv


def plot_iv_surface(
    S0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Save a 3D implied volatility surface plot."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    M, T = np.meshgrid(strikes / S0, maturities)
    fig = plt.figure(figsize=(9.5, 6.0), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(M, T, iv * 100.0, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(title, pad=16)
    ax.set_xlabel("Moneyness K/S0")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("Implied volatility (%)")
    fig.colorbar(surf, shrink=0.55, aspect=12, pad=0.1)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_smiles(
    S0: float,
    strikes: np.ndarray,
    maturity: float,
    r: float,
    q: float,
    scenarios: Dict[str, HestonParams],
    title: str,
    filename: str,
) -> None:
    """Save smile comparison for several Heston parameter scenarios."""
    fig = plt.figure(figsize=(8.8, 5.2), dpi=160)
    for label, params in scenarios.items():
        iv = implied_vol_surface(S0, strikes, np.array([maturity]), r, q, params, N=4096, eta=0.25)[0]
        plt.plot(strikes / S0, iv * 100.0, marker="o", linewidth=1.7, label=label)
    plt.title(title)
    plt.xlabel("Moneyness K/S0")
    plt.ylabel("Implied volatility (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def representative_market_quotes(
    S0: float,
    r: float,
    q: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a representative liquid option quote grid.

    In production, replace this block with exchange/OTC mid quotes converted to implied vols.
    The table below is generated from a Heston model plus small bid/ask-like perturbations.
    """
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    maturities = np.array([0.25, 0.50, 1.00, 2.00], dtype=float)
    true_params = HestonParams(v0=0.0400, kappa=1.70, theta=0.0450, xi=0.55, rho=-0.70)
    clean_iv = implied_vol_surface(S0, strikes, maturities, r, q, true_params, N=4096, eta=0.25)

    perturbation = np.array([
        [0.0020, 0.0015, 0.0000, -0.0005, -0.0010],
        [0.0015, 0.0010, 0.0000, -0.0007, -0.0010],
        [0.0010, 0.0005, 0.0000, -0.0005, -0.0008],
        [0.0005, 0.0002, 0.0000, -0.0003, -0.0005],
    ])
    market_iv = np.maximum(clean_iv + perturbation, 0.01)

    market_prices = np.empty_like(market_iv)
    for ti, T in enumerate(maturities):
        for ki, K in enumerate(strikes):
            market_prices[ti, ki] = black_scholes_call(S0, K, T, r, q, market_iv[ti, ki])
    return strikes, maturities, market_iv, market_prices


def calibrate_heston_to_quotes(
    S0: float,
    r: float,
    q: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_iv: np.ndarray,
    market_prices: np.ndarray,
) -> Tuple[HestonParams, float, np.ndarray, np.ndarray]:
    """
    Calibrate Heston parameters by minimizing vega-normalized price errors.

    This objective behaves similarly to minimizing implied-vol errors while avoiding
    an implied-vol inversion inside every optimizer iteration.
    """
    vegas = np.empty_like(market_prices)
    for ti, T in enumerate(maturities):
        for ki, K in enumerate(strikes):
            vegas[ti, ki] = max(bs_vega(S0, K, T, r, q, market_iv[ti, ki]), 1.0)

    def unpack(x: np.ndarray) -> HestonParams:
        return HestonParams(v0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])

    def objective(x: np.ndarray) -> float:
        params = unpack(x)
        # Soft Feller penalty. We do not hard-enforce it because real calibrations
        # often violate 2*kappa*theta > xi^2 while still fitting option smiles well.
        feller_gap = 2.0 * params.kappa * params.theta - params.xi ** 2
        penalty = 0.01 * max(-feller_gap, 0.0) ** 2

        error_terms = []
        for ti, T in enumerate(maturities):
            model_prices = call_prices_for_strikes(S0, strikes, T, r, q, params, N=1024, eta=0.25)
            error_terms.extend(((model_prices - market_prices[ti, :]) / vegas[ti, :]).tolist())
        return float(np.mean(np.square(error_terms)) + penalty)

    x0 = np.array([0.0350, 1.00, 0.0400, 0.45, -0.50])
    bounds = [(0.005, 0.250), (0.05, 8.00), (0.005, 0.250), (0.05, 2.00), (-0.95, 0.20)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 150, "ftol": 1.0e-10})
    calibrated = unpack(result.x)

    model_iv = implied_vol_surface(S0, strikes, maturities, r, q, calibrated, N=4096, eta=0.25)
    model_prices = np.empty_like(model_iv)
    for ti, T in enumerate(maturities):
        model_prices[ti, :] = call_prices_for_strikes(S0, strikes, T, r, q, calibrated, N=4096, eta=0.25)
    rmse_iv = float(np.sqrt(np.nanmean((model_iv - market_iv) ** 2)))
    return calibrated, rmse_iv, model_iv, model_prices


def plot_calibration_fit(
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_iv: np.ndarray,
    model_iv: np.ndarray,
    filename: str,
) -> None:
    """Save market vs model smile plot by maturity."""
    fig = plt.figure(figsize=(9.0, 5.4), dpi=160)
    for ti, T in enumerate(maturities):
        plt.plot(strikes, market_iv[ti] * 100.0, marker="o", linestyle="", label=f"Market T={T:.2f}y")
        plt.plot(strikes, model_iv[ti] * 100.0, linewidth=1.6, label=f"Heston fit T={T:.2f}y")
    plt.title("Calibration fit: market implied vol quotes vs Heston model")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def heston_greeks_fd(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    dS: float = 0.01,
    dv: float = 1.0e-4,
    dr: float = 1.0e-4,
) -> Dict[str, float]:
    """Finite-difference Greeks for a Heston European call."""
    base = heston_call_fft(S0, K, T, r, q, p)
    up_s = heston_call_fft(S0 + dS, K, T, r, q, p)
    down_s = heston_call_fft(S0 - dS, K, T, r, q, p)

    p_up_v0 = HestonParams(v0=p.v0 + dv, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho)
    p_down_v0 = HestonParams(v0=max(p.v0 - dv, 1.0e-8), kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho)
    up_v = heston_call_fft(S0, K, T, r, q, p_up_v0)
    down_v = heston_call_fft(S0, K, T, r, q, p_down_v0)

    up_r = heston_call_fft(S0, K, T, r + dr, q, p)
    down_r = heston_call_fft(S0, K, T, r - dr, q, p)

    up_t = heston_call_fft(S0, K, max(T - 1.0 / 365.0, 1.0e-6), r, q, p)

    return {
        "price": base,
        "delta": (up_s - down_s) / (2.0 * dS),
        "gamma": (up_s - 2.0 * base + down_s) / (dS ** 2),
        "vega_v0": (up_v - down_v) / (2.0 * dv),
    
        "rho_rate": (up_r - down_r) / (2.0 * dr),
        "theta_per_day": up_t - base,
    }


def main() -> None:
    S0 = 100.0
    r = 0.045
    q = 0.0
    base = HestonParams(v0=0.0400, kappa=1.50, theta=0.0400, xi=0.50, rho=-0.65)

    strikes = np.linspace(70, 130, 13)
    maturities = np.array([0.10, 0.25, 0.50, 1.00, 2.00, 3.00])

    # 1. Pricing sanity check: Fourier inverse vs FFT at one point.
    K = 50.0
    T = 1.0
    call_inverse = heston_call_fourier_inverse(S0, K, T, r, q, base)
    call_fft = heston_call_fft(S0, K, T, r, q, base)
    put_fft = heston_put_fft(S0, K, T, r, q, base)
    print("Pricing check")
    print(f"Call by Fourier inverse: {call_inverse:,.6f}")
    print(f"Call by FFT interpolation: {call_fft:,.6f}")
    print(f"Put by put-call parity: {put_fft:,.6f}")

    # 2. Baseline implied volatility surface.
    base_iv = implied_vol_surface(S0, strikes, maturities, r, q, base, N=4096, eta=0.25)
    plot_iv_surface(S0, strikes, maturities, base_iv, "Baseline Heston implied volatility surface", "figures/heston_baseline_surface.png")

    # 3. Parameter effects: smiles and surfaces.
    scenarios_rho = {
        "rho=-0.85": HestonParams(base.v0, base.kappa, base.theta, base.xi, -0.85),
        "rho=-0.40": HestonParams(base.v0, base.kappa, base.theta, base.xi, -0.40),
        "rho=0.00": HestonParams(base.v0, base.kappa, base.theta, base.xi, 0.00),
    }
    plot_parameter_smiles(S0, strikes, 1.0, r, q, scenarios_rho, "Effect of rho on 1Y implied-volatility skew", "figures/rho_smile_effect.png")

    scenarios_xi = {
        "xi=0.25": HestonParams(base.v0, base.kappa, base.theta, 0.25, base.rho),
        "xi=0.50": HestonParams(base.v0, base.kappa, base.theta, 0.50, base.rho),
        "xi=0.90": HestonParams(base.v0, base.kappa, base.theta, 0.90, base.rho),
    }
    plot_parameter_smiles(S0, strikes, 1.0, r, q, scenarios_xi, "Effect of vol-of-vol xi on 1Y smile curvature", "figures/xi_smile_effect.png")

    altered_surfaces = {
        "High vol-of-vol xi=0.90": HestonParams(base.v0, base.kappa, base.theta, 0.90, base.rho),
        "More negative rho=-0.85": HestonParams(base.v0, base.kappa, base.theta, base.xi, -0.85),
        "High initial variance v0=0.09": HestonParams(0.0900, base.kappa, base.theta, base.xi, base.rho),
        "High long-run variance theta=0.09": HestonParams(base.v0, base.kappa, 0.0900, base.xi, base.rho),
    }
    for title, params in altered_surfaces.items():
        iv = implied_vol_surface(S0, strikes, maturities, r, q, params, N=4096, eta=0.25)
        safe_title = title.lower().replace(" ", "_").replace("=", "_").replace("-", "m").replace(".", "p")
        plot_iv_surface(S0, strikes, maturities, iv, title, f"figures/{safe_title}.png")

    # 4. End-to-end calibration to representative liquid option quotes.
    quote_strikes, quote_maturities, market_iv, market_prices = representative_market_quotes(S0, r, q)
    calibrated, rmse_iv, model_iv, model_prices = calibrate_heston_to_quotes(
        S0, r, q, quote_strikes, quote_maturities, market_iv, market_prices
    )
    plot_calibration_fit(quote_strikes, quote_maturities, market_iv, model_iv, "figures/calibration_fit.png")

    print("\nCalibration result")
    print(calibrated)
    print(f"Implied-vol RMSE: {rmse_iv * 100.0:.4f} vol points")
    print("\nMarket implied vol matrix")
    print(np.round(market_iv * 100.0, 4))
    print("\nModel implied vol matrix")
    print(np.round(model_iv * 100.0, 4))

    # 5. Greeks at the calibrated parameter set.
    greeks = heston_greeks_fd(S0, 100.0, 1.0, r, q, calibrated)
    print("\nFinite-difference Greeks for 1Y ATM call")
    for name, value in greeks.items():
        print(f"{name:>14s}: {value:,.6f}")

    # Export tables for inclusion in the PDF.
    with open("calibration_summary.txt", "w", encoding="utf-8") as f:
        f.write("Calibrated Heston parameters\n")
        f.write(f"v0={calibrated.v0:.6f}\n")
        f.write(f"kappa={calibrated.kappa:.6f}\n")
        f.write(f"theta={calibrated.theta:.6f}\n")
        f.write(f"xi={calibrated.xi:.6f}\n")
        f.write(f"rho={calibrated.rho:.6f}\n")
        f.write(f"iv_rmse_vol_points={rmse_iv*100.0:.6f}\n")
        f.write("\nGreeks\n")
        for name, value in greeks.items():
            f.write(f"{name}={value:.8f}\n")


if __name__ == "__main__":
    main()
