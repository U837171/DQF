"""Peaks2Tails Math Primer - Econometrics examples.

This file is intentionally self-contained for students. Put this `.py` file
and the matching CSV files in the same local folder, then run:

    python econometrics_primer.py

Required CSV files:
    econometrics_price_path.csv
    econometrics_factor_returns.csv
    econometrics_garch_inputs.csv
    econometrics_cointegration_series.csv

The examples use only pandas, numpy, and matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

PRICE_FILE = "econometrics_price_path.csv"
FACTOR_FILE = "econometrics_factor_returns.csv"
GARCH_FILE = "econometrics_garch_inputs.csv"
COINTEGRATION_FILE = "econometrics_cointegration_series.csv"


def read_csv_here(file_name: str) -> pd.DataFrame:
    """Read a CSV located beside this script."""
    return pd.read_csv(HERE / file_name)


def arithmetic_returns(prices: pd.Series) -> pd.Series:
    """Arithmetic returns: r_t = S_t / S_{t-1} - 1."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns: ell_t = log(S_t / S_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def sample_mean(x: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=float)))


def sample_volatility(x: pd.Series | np.ndarray, ddof: int = 1) -> float:
    return float(np.std(np.asarray(x, dtype=float), ddof=ddof))


def annualize_volatility(daily_volatility: float, trading_days: int = TRADING_DAYS) -> float:
    """Square-root-of-time annualization."""
    return daily_volatility * sqrt(trading_days)


def normal_var_from_pnl_mean_vol(mean_pnl: float, vol_pnl: float, z_alpha: float) -> float:
    """Parametric VaR for loss L = -PnL when PnL is normal."""
    return -mean_pnl + z_alpha * vol_pnl


def expected_exceptions(sample_size: int, exception_probability: float) -> float:
    return sample_size * exception_probability


def binomial_probability(n: int, k: int, p: float) -> float:
    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def binomial_tail_probability(n: int, k_min: int, p: float) -> float:
    return sum(binomial_probability(n, k, p) for k in range(k_min, n + 1))


@dataclass(frozen=True)
class OLSResult:
    alpha: float
    betas: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    r_squared: float
    residual_volatility: float


def ols_fit(y: pd.Series | np.ndarray, x: pd.DataFrame | np.ndarray, add_intercept: bool = True) -> OLSResult:
    """OLS using the normal equations.

    y is the dependent variable.
    x contains one or more regressors.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if add_intercept:
        design = np.column_stack([np.ones(len(x_arr)), x_arr])
    else:
        design = x_arr

    beta_hat = np.linalg.solve(design.T @ design, design.T @ y_arr).ravel()
    fitted = design @ beta_hat
    residuals = y_arr.ravel() - fitted
    tss = float(np.sum((y_arr.ravel() - y_arr.mean()) ** 2))
    rss = float(np.sum(residuals**2))
    r_squared = 1.0 - rss / tss if tss > 0 else np.nan
    residual_vol = sample_volatility(residuals)

    if add_intercept:
        alpha = float(beta_hat[0])
        betas = beta_hat[1:]
    else:
        alpha = 0.0
        betas = beta_hat

    return OLSResult(
        alpha=alpha,
        betas=betas,
        residuals=residuals,
        fitted=fitted,
        r_squared=float(r_squared),
        residual_volatility=residual_vol,
    )


def systematic_residual_variance(beta: float, factor_vol: float, residual_vol: float) -> tuple[float, float, float]:
    """Variance decomposition for a one-factor regression."""
    systematic = beta**2 * factor_vol**2
    residual = residual_vol**2
    return systematic, residual, systematic + residual


def ar1_moments(c: float, phi: float, innovation_volatility: float) -> tuple[float, float]:
    """Stationary AR(1) mean and volatility."""
    if abs(phi) >= 1:
        raise ValueError("Stationary AR(1) moments require abs(phi) < 1.")
    mean = c / (1.0 - phi)
    variance = innovation_volatility**2 / (1.0 - phi**2)
    return mean, sqrt(variance)


def ar1_forecast(c: float, phi: float, x_t: float) -> float:
    return c + phi * x_t


def ar1_autocorrelation(phi: float, max_lag: int) -> pd.DataFrame:
    lags = np.arange(1, max_lag + 1)
    return pd.DataFrame({"lag": lags, "acf": phi**lags})


def estimate_ar1(series: pd.Series) -> tuple[float, float, float]:
    """Estimate X_t = c + phi X_{t-1} + epsilon_t by OLS."""
    y = series.iloc[1:].to_numpy()
    x_lag = series.shift(1).dropna().to_numpy()
    fit = ols_fit(y, x_lag)
    c = fit.alpha
    phi = float(fit.betas[0])
    innovation_vol = sample_volatility(fit.residuals)
    return c, phi, innovation_vol


def ewma_variance(returns: pd.Series, decay: float, initial_variance: float | None = None) -> pd.Series:
    """EWMA variance: sigma_{t+1}^2 = lambda sigma_t^2 + (1-lambda) r_t^2."""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be between 0 and 1.")
    r = returns.astype(float).reset_index(drop=True)
    sigma2 = [float(initial_variance if initial_variance is not None else r.var(ddof=1))]
    for value in r:
        sigma2.append(decay * sigma2[-1] + (1.0 - decay) * value**2)
    return pd.Series(sigma2[1:], name="ewma_variance")


def garch_11_variance(
    returns: pd.Series,
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float | None = None,
) -> pd.Series:
    """GARCH(1,1) conditional variance path."""
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("GARCH parameters require omega > 0, alpha >= 0, beta >= 0.")
    if alpha + beta >= 1:
        raise ValueError("Covariance-stationary GARCH(1,1) requires alpha + beta < 1.")

    r = returns.astype(float).reset_index(drop=True)
    sigma2 = [float(initial_variance if initial_variance is not None else omega / (1.0 - alpha - beta))]
    for value in r:
        sigma2.append(omega + alpha * value**2 + beta * sigma2[-1])
    return pd.Series(sigma2[1:], name="garch_variance")


def garch_long_run_volatility(omega: float, alpha: float, beta: float) -> float:
    if alpha + beta >= 1:
        raise ValueError("Long-run volatility exists only when alpha + beta < 1.")
    return sqrt(omega / (1.0 - alpha - beta))


def residual_autocorrelation(residuals: np.ndarray, lag: int = 1) -> float:
    if lag <= 0 or lag >= len(residuals):
        raise ValueError("lag must be positive and smaller than the residual length.")
    x = residuals[:-lag]
    y = residuals[lag:]
    return float(np.corrcoef(x, y)[0, 1])


def hedge_ratio_from_regression(portfolio_pnl: pd.Series, hedge_pnl: pd.Series) -> float:
    """Minimum-variance hedge ratio for P - hH."""
    covariance = float(np.cov(portfolio_pnl, hedge_pnl, ddof=1)[0, 1])
    variance = float(np.var(hedge_pnl, ddof=1))
    return covariance / variance


def cointegration_residual(y: pd.Series, x: pd.Series) -> tuple[float, pd.Series]:
    """Estimate Y_t = a + theta X_t + residual_t and return theta and residual."""
    fit = ols_fit(y, x)
    theta = float(fit.betas[0])
    residual = pd.Series(fit.residuals, index=y.index, name="cointegration_residual")
    return theta, residual


def pyplot_for_png():
    """Load matplotlib with a file-saving backend.

    Important: this function is used by the `.py` script only when saving PNGs.
    The module does not force an Agg backend at import time, so notebooks keep
    their normal inline display backend.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def save_return_plot(price_df: pd.DataFrame, output_path: Path) -> None:
    plt = pyplot_for_png()
    if plt is None:
        print(f"Skipped plot {output_path.name}: matplotlib is not installed.")
        return
    returns = arithmetic_returns(price_df["price"])
    colors = ["#B00000" if r < 0 else "#1A6599" for r in returns]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.bar(price_df["date"].iloc[1:], returns * 100, color=colors, alpha=0.88)
    ax.axhline(0, color="#22252A", linewidth=0.8)
    ax.set_title("Arithmetic returns from price path")
    ax.set_ylabel("Return (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_regression_plot(factor_df: pd.DataFrame, fit: OLSResult, output_path: Path) -> None:
    plt = pyplot_for_png()
    if plt is None:
        print(f"Skipped plot {output_path.name}: matplotlib is not installed.")
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = factor_df["market_return"].to_numpy()
    y = factor_df["portfolio_return"].to_numpy()
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = fit.alpha + fit.betas[0] * x_line
    ax.scatter(x * 100, y * 100, color="#1A6599", label="observations")
    ax.plot(x_line * 100, y_line * 100, color="#B00000", label="OLS fit")
    ax.set_title("Portfolio return beta regression")
    ax.set_xlabel("Market return (%)")
    ax.set_ylabel("Portfolio return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_two_factor_regression_plane(factor_df: pd.DataFrame, fit: OLSResult, output_path: Path) -> None:
    plt = pyplot_for_png()
    if plt is None:
        print(f"Skipped plot {output_path.name}: matplotlib is not installed.")
        return
    fig = plt.figure(figsize=(7.2, 5.0), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    x = factor_df["market_return"].to_numpy()
    y = factor_df["rate_change"].to_numpy()
    z = factor_df["portfolio_return"].to_numpy()
    x_grid, y_grid = np.meshgrid(
        np.linspace(x.min(), x.max(), 18),
        np.linspace(y.min(), y.max(), 18),
    )
    z_grid = fit.alpha + fit.betas[0] * x_grid + fit.betas[1] * y_grid

    ax.scatter(x * 100, y * 10000, z * 100, color="#B00000", s=28, depthshade=True)
    ax.plot_surface(
        x_grid * 100,
        y_grid * 10000,
        z_grid * 100,
        cmap="viridis",
        alpha=0.72,
        linewidth=0,
        antialiased=True,
    )
    ax.set_title("Two-factor regression plane")
    ax.set_xlabel("Market return (%)")
    ax.set_ylabel("Rate move (bps)")
    ax.set_zlabel("Portfolio return (%)")
    ax.view_init(elev=23, azim=-135)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_volatility_plot(garch_df: pd.DataFrame, ewma_var: pd.Series, garch_var: pd.Series, output_path: Path) -> None:
    plt = pyplot_for_png()
    if plt is None:
        print(f"Skipped plot {output_path.name}: matplotlib is not installed.")
        return
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    ax.plot(garch_df["date"], np.sqrt(ewma_var) * 100, color="#1A6599", label="EWMA volatility")
    ax.plot(garch_df["date"], np.sqrt(garch_var) * 100, color="#2D7A47", label="GARCH volatility")
    ax.set_title("Conditional volatility estimates")
    ax.set_ylabel("Volatility (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_cointegration_plot(cointegration_df: pd.DataFrame, residual: pd.Series, output_path: Path) -> None:
    plt = pyplot_for_png()
    if plt is None:
        print(f"Skipped plot {output_path.name}: matplotlib is not installed.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), dpi=150, sharex=True)
    axes[0].plot(cointegration_df["date"], cointegration_df["series_x"], color="#1A6599", label="Series X")
    axes[0].plot(cointegration_df["date"], cointegration_df["series_y"], color="#F0A202", label="Series Y")
    axes[0].set_title("Price-level series")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(cointegration_df["date"], residual, color="#B00000", label="Estimated spread")
    axes[1].axhline(0, color="#22252A", linewidth=0.8)
    axes[1].set_title("Regression residual")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def case_study_returns() -> None:
    print("\nCase Study 1: Returns, volatility, and normal VaR")
    price_df = read_csv_here(PRICE_FILE)
    arith = arithmetic_returns(price_df["price"])
    logs = log_returns(price_df["price"])

    daily_mean = sample_mean(arith)
    daily_vol = sample_volatility(arith)
    annual_vol = annualize_volatility(daily_vol)
    var_99 = normal_var_from_pnl_mean_vol(daily_mean, daily_vol, z_alpha=2.326)

    summary = pd.DataFrame(
        {
            "date": price_df["date"].iloc[1:].to_numpy(),
            "arithmetic_return": arith.to_numpy(),
            "log_return": logs.to_numpy(),
        }
    )
    print(summary.round(6).to_string(index=False))
    print(f"Daily mean return: {daily_mean:.6f}")
    print(f"Daily sample volatility: {daily_vol:.6f}")
    print(f"Annualized volatility: {annual_vol:.6f}")
    print(f"99 percent one-day normal VaR on loss return: {var_99:.6f}")
    save_return_plot(price_df, HERE / "econometrics_primer_returns_plot.png")


def case_study_factor_regression() -> None:
    print("\nCase Study 2: Factor regression and variance decomposition")
    factor_df = read_csv_here(FACTOR_FILE)
    fit = ols_fit(
        y=factor_df["portfolio_return"],
        x=factor_df[["market_return", "rate_change"]],
    )
    print(f"Alpha: {fit.alpha:.6f}")
    print(f"Market beta: {fit.betas[0]:.6f}")
    print(f"Rate beta: {fit.betas[1]:.6f}")
    print(f"R-squared: {fit.r_squared:.4f}")
    print(f"Residual volatility: {fit.residual_volatility:.6f}")

    one_factor = ols_fit(factor_df["portfolio_return"], factor_df["market_return"])
    sys_var, resid_var, total_var = systematic_residual_variance(
        beta=float(one_factor.betas[0]),
        factor_vol=sample_volatility(factor_df["market_return"]),
        residual_vol=one_factor.residual_volatility,
    )
    print("One-factor variance decomposition:")
    print(f"  systematic variance: {sys_var:.8f}")
    print(f"  residual variance:   {resid_var:.8f}")
    print(f"  total variance:      {total_var:.8f}")
    save_regression_plot(factor_df, one_factor, HERE / "econometrics_primer_regression_plot.png")
    save_two_factor_regression_plane(
        factor_df,
        fit,
        HERE / "econometrics_primer_two_factor_regression_plane.png",
    )


def case_study_time_series() -> None:
    print("\nCase Study 3: AR(1), EWMA, and GARCH volatility")
    garch_df = read_csv_here(GARCH_FILE)
    returns = garch_df["return"]
    c, phi, innovation_vol = estimate_ar1(returns)
    print(f"Estimated AR(1) c: {c:.6f}")
    print(f"Estimated AR(1) phi: {phi:.6f}")
    print(f"Estimated innovation volatility: {innovation_vol:.6f}")
    if abs(phi) < 1:
        mean, vol = ar1_moments(c, phi, innovation_vol)
        print(f"Estimated stationary mean: {mean:.6f}")
        print(f"Estimated stationary volatility: {vol:.6f}")
    print(f"Next return forecast from last value: {ar1_forecast(c, phi, returns.iloc[-1]):.6f}")

    acf = ar1_autocorrelation(phi=0.72, max_lag=10)
    print("\nTheoretical AR(1) ACF with phi=0.72:")
    print(acf.round(4).to_string(index=False))

    ewma_var = ewma_variance(returns, decay=0.94)
    garch_var = garch_11_variance(returns, omega=0.000002, alpha=0.08, beta=0.90)
    print(f"Latest EWMA volatility: {sqrt(float(ewma_var.iloc[-1])):.6f}")
    print(f"Latest GARCH volatility: {sqrt(float(garch_var.iloc[-1])):.6f}")
    print(f"GARCH long-run volatility: {garch_long_run_volatility(0.000002, 0.08, 0.90):.6f}")
    save_volatility_plot(garch_df, ewma_var, garch_var, HERE / "econometrics_primer_volatility_plot.png")


def case_study_backtesting_and_cointegration() -> None:
    print("\nCase Study 4: VaR exceptions and cointegration residuals")
    expected = expected_exceptions(sample_size=250, exception_probability=0.01)
    prob_exact_8 = binomial_probability(n=250, k=8, p=0.01)
    prob_8_or_more = binomial_tail_probability(n=250, k_min=8, p=0.01)
    print(f"Expected 99 percent VaR exceptions in 250 days: {expected:.2f}")
    print(f"P(exactly 8 exceptions): {prob_exact_8:.6f}")
    print(f"P(8 or more exceptions): {prob_8_or_more:.6f}")

    cointegration_df = read_csv_here(COINTEGRATION_FILE)
    theta, residual = cointegration_residual(cointegration_df["series_y"], cointegration_df["series_x"])
    c, phi, innovation_vol = estimate_ar1(residual)
    print(f"Estimated hedge ratio theta in Y = a + theta X + residual: {theta:.6f}")
    print(f"Residual AR(1) phi: {phi:.6f}")
    print(f"Residual innovation volatility: {innovation_vol:.6f}")
    print("A residual phi below 1 is consistent with mean reversion, but it is not a formal unit-root test.")
    save_cointegration_plot(
        cointegration_df,
        residual,
        HERE / "econometrics_primer_cointegration_plot.png",
    )


def main() -> None:
    case_study_returns()
    case_study_factor_regression()
    case_study_time_series()
    case_study_backtesting_and_cointegration()
    print("\nPlots saved in the same folder as this script.")


if __name__ == "__main__":
    main()
