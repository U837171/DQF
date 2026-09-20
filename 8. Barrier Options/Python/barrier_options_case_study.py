
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PARAMS_FILE = "barrier_options_params.csv"
MARKET_FILE = "barrier_options_market.csv"


def norm_cdf(x):
    x_arr = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    out = 0.5 * (1.0 + erf_vec(x_arr / math.sqrt(2.0)))
    if np.ndim(x) == 0:
        return float(out)
    return out


def black_scholes_call(S, K, r, q, sigma, T):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def simulate_gbm(S0, r, q, sigma, T, steps, paths, seed=7):
    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.standard_normal((paths, steps))
    increments = (r - q - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(np.column_stack([np.full(paths, math.log(S0)), increments]), axis=1)
    log_paths[:, 1:] = log_paths[:, 1:]
    return np.exp(log_paths)


def case_study():
    params = pd.read_csv(PARAMS_FILE).set_index("parameter")["value"].astype(float)
    market = pd.read_csv(MARKET_FILE)
    S0 = float(params.get("S0", 100.0))
    K = float(params.get("K", 100.0))
    r = float(params.get("r", 0.04))
    q = float(params.get("q", 0.01))
    sigma = float(params.get("sigma", 0.22))
    T = float(params.get("T", 1.0))
    steps = int(params.get("steps", 64))
    paths = int(params.get("paths", 12000))
    paths_array = simulate_gbm(S0, r, q, sigma, T, steps, paths)
    european = black_scholes_call(S0, K, r, q, sigma, T)
    result = run_topic_case("barrier_mc_bridge", params, market, paths_array)
    result["bsm_call_reference"] = european
    return result



def case_forward_swap_sensitivities(params, market, paths_array):
    df = market.copy()
    df["discount_factor"] = np.exp(-df["zero_rate"] * df["tenor"])
    annuity = float((df["accrual"] * df["discount_factor"]).sum())
    par_swap = float((1.0 - df["discount_factor"].iloc[-1]) / annuity)
    bump = 0.0001
    bumped_df = np.exp(-(df["zero_rate"] + bump) * df["tenor"])
    bumped_annuity = float((df["accrual"] * bumped_df).sum())
    bumped_par = float((1.0 - bumped_df.iloc[-1]) / bumped_annuity)
    return {"curve_table": df, "par_swap_rate": par_swap, "par_rate_dv01": (bumped_par - par_swap) / bump}


def case_fx_forwards_options(params, market, paths_array):
    df = market.copy()
    df["fx_forward"] = df["spot"] * np.exp((df["rd"] - df["rf"]) * df["tenor"])
    df["forward_points"] = df["fx_forward"] - df["spot"]
    return {"fx_forward_table": df, "mean_forward_points": float(df["forward_points"].mean())}


def case_delta_hedging_error(params, market, paths_array):
    S0, K, r, q, sigma, T = [float(params[x]) for x in ["S0", "K", "r", "q", "sigma", "T"]]
    eps = 0.01
    price = black_scholes_call(S0, K, r, q, sigma, T)
    delta = (black_scholes_call(S0 + eps, K, r, q, sigma, T) - black_scholes_call(S0 - eps, K, r, q, sigma, T)) / (2 * eps)
    gamma = (black_scholes_call(S0 + eps, K, r, q, sigma, T) - 2 * price + black_scholes_call(S0 - eps, K, r, q, sigma, T)) / (eps ** 2)
    shocks = market["shock"].to_numpy()
    pnl_approx = delta * shocks + 0.5 * gamma * shocks ** 2
    return {"delta": delta, "gamma": gamma, "hedge_pnl_table": pd.DataFrame({"shock": shocks, "pnl_approx": pnl_approx})}


def case_ois_curve_bootstrap(params, market, paths_array):
    df = market.copy()
    df["discount_factor"] = 1.0 / (1.0 + df["zero_rate"] * df["tenor"])
    df["forward_rate"] = -np.log(df["discount_factor"] / df["discount_factor"].shift(1).fillna(1.0)) / df["tenor"].diff().fillna(df["tenor"])
    return {"bootstrapped_curve": df, "terminal_discount_factor": float(df["discount_factor"].iloc[-1])}


def case_factor_portfolio_optimization(params, market, paths_array):
    returns = market[["asset_1", "asset_2", "asset_3"]].to_numpy()
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T)
    inv = np.linalg.pinv(cov)
    ones = np.ones(3)
    min_var_w = inv @ ones / (ones @ inv @ ones)
    return {"mean_returns": mu, "covariance": cov, "minimum_variance_weights": min_var_w}


def case_brownian_girsanov(params, market, paths_array):
    theta = float(params.get("theta", 0.35))
    T = float(params.get("T", 1.0))
    W_T = np.log(paths_array[:, -1] / paths_array[:, 0])
    Z_T = np.exp(-theta * W_T - 0.5 * theta * theta * T)
    return {"rn_density_mean": float(Z_T.mean()), "weighted_terminal_log_return": float(np.mean(Z_T * W_T)), "density_std": float(Z_T.std(ddof=1))}


def case_measure_change_tree(params, market, paths_array):
    df = market.copy()
    df["zeta"] = df["q_prob"] / df["p_prob"]
    df["q_from_p"] = df["p_prob"] * df["zeta"]
    df["claim_weighted_q"] = df["q_from_p"] * df["claim"]
    return {"rn_tree": df, "q_expectation": float(df["claim_weighted_q"].sum()), "zeta_p_mean": float((df["p_prob"] * df["zeta"]).sum())}


def case_bsm_greeks_pde(params, market, paths_array):
    S0, K, r, q, sigma, T = [float(params[x]) for x in ["S0", "K", "r", "q", "sigma", "T"]]
    h = 0.05
    price = black_scholes_call(S0, K, r, q, sigma, T)
    delta = (black_scholes_call(S0 + h, K, r, q, sigma, T) - black_scholes_call(S0 - h, K, r, q, sigma, T)) / (2*h)
    gamma = (black_scholes_call(S0 + h, K, r, q, sigma, T) - 2*price + black_scholes_call(S0 - h, K, r, q, sigma, T)) / h**2
    theta = (black_scholes_call(S0, K, r, q, sigma, T - 1/365) - price) / (-1/365)
    residual = theta + 0.5*sigma*sigma*S0*S0*gamma + r*S0*delta - r*price
    return {"price": price, "delta": delta, "gamma": gamma, "pde_residual": residual}


def case_american_lsm(params, market, paths_array):
    K = float(params.get("K", 100.0))
    r = float(params.get("r", 0.04))
    T = float(params.get("T", 1.0))
    steps = paths_array.shape[1] - 1
    dt = T / steps
    cashflow = np.maximum(K - paths_array[:, -1], 0.0)
    exercise_time = np.full(paths_array.shape[0], steps)
    for t in range(steps - 1, 0, -1):
        itm = K > paths_array[:, t]
        if itm.sum() < 5:
            cashflow *= math.exp(-r * dt)
            continue
        x = paths_array[itm, t]
        y = cashflow[itm] * math.exp(-r * dt)
        X = np.column_stack([np.ones_like(x), x, x*x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        continuation = X @ beta
        exercise = K - x
        do_ex = exercise > continuation
        idx = np.where(itm)[0][do_ex]
        cashflow[idx] = exercise[do_ex]
        exercise_time[idx] = t
        cashflow[~np.isin(np.arange(len(cashflow)), idx)] *= math.exp(-r * dt)
    return {"lsm_american_put": float(cashflow.mean() * math.exp(-r * dt)), "mean_exercise_step": float(exercise_time.mean())}


def case_barrier_mc_bridge(params, market, paths_array):
    K = float(params.get("K", 100.0))
    H = float(params.get("barrier", 80.0))
    r = float(params.get("r", 0.04))
    T = float(params.get("T", 1.0))
    knocked = (paths_array.min(axis=1) <= H)
    payoff = np.where(knocked, 0.0, np.maximum(paths_array[:, -1] - K, 0.0))
    return {"down_out_call_mc": float(math.exp(-r*T) * payoff.mean()), "knockout_probability_discrete": float(knocked.mean())}


def case_asian_mc_control_variate(params, market, paths_array):
    K = float(params.get("K", 100.0))
    r = float(params.get("r", 0.04))
    T = float(params.get("T", 1.0))
    arith = paths_array[:, 1:].mean(axis=1)
    geom = np.exp(np.log(paths_array[:, 1:]).mean(axis=1))
    arith_payoff = np.maximum(arith - K, 0.0)
    geom_payoff = np.maximum(geom - K, 0.0)
    beta = np.cov(arith_payoff, geom_payoff)[0, 1] / np.var(geom_payoff)
    cv = arith_payoff - beta * (geom_payoff - geom_payoff.mean())
    return {"asian_arithmetic_mc": float(math.exp(-r*T)*arith_payoff.mean()), "asian_control_variate_mc": float(math.exp(-r*T)*cv.mean()), "control_beta": float(beta)}


def case_svi_surface(params, market, paths_array):
    k = market["log_moneyness"].to_numpy()
    a, b, rho, m, sig = [float(params.get(x, d)) for x, d in [("a", 0.02), ("b", 0.15), ("rho", -0.35), ("m", 0.0), ("svi_sigma", 0.25)]]
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sig*sig))
    return {"svi_total_variance": pd.DataFrame({"log_moneyness": k, "total_variance": w, "implied_vol": np.sqrt(w / params.get("T", 1.0))})}


def case_merton_jump_diffusion(params, market, paths_array):
    lam = float(params.get("lambda", 0.7))
    muj = float(params.get("jump_mean", -0.08))
    sigj = float(params.get("jump_vol", 0.18))
    rng = np.random.default_rng(99)
    jumps = rng.poisson(lam * params.get("T", 1.0), size=paths_array.shape[0])
    jump_sizes = rng.normal(muj * jumps, sigj * np.sqrt(np.maximum(jumps, 1e-12)))
    terminal = paths_array[:, -1] * np.exp(jump_sizes)
    payoff = np.maximum(terminal - params.get("K", 100.0), 0.0)
    return {"merton_mc_call": float(math.exp(-params.get("r", 0.04)*params.get("T", 1.0))*payoff.mean()), "mean_jump_count": float(jumps.mean())}


def case_fourier_characteristic_function(params, market, paths_array):
    u = market["u"].to_numpy()
    mu = math.log(params.get("S0", 100.0)) + (params.get("r", 0.04) - params.get("q", 0.01) - 0.5*params.get("sigma", 0.22)**2)*params.get("T", 1.0)
    var = params.get("sigma", 0.22)**2 * params.get("T", 1.0)
    phi_real = np.exp(-0.5 * var * u*u) * np.cos(u * mu)
    phi_imag = np.exp(-0.5 * var * u*u) * np.sin(u * mu)
    return {"characteristic_function": pd.DataFrame({"u": u, "real": phi_real, "imag": phi_imag})}


def case_mc_variance_reduction_lsm(params, market, paths_array):
    K = float(params.get("K", 100.0))
    r = float(params.get("r", 0.04))
    T = float(params.get("T", 1.0))
    payoff = np.maximum(paths_array[:, -1] - K, 0.0)
    antithetic_estimate = math.exp(-r*T) * payoff.mean()
    se = math.exp(-r*T) * payoff.std(ddof=1) / math.sqrt(len(payoff))
    return {"mc_call_estimate": float(antithetic_estimate), "standard_error": float(se), "paths": len(payoff)}



def run_topic_case(case_name, params, market, paths_array):
    return CASE_DISPATCH[case_name](params, market, paths_array)


CASE_DISPATCH = {
    "barrier_mc_bridge": case_barrier_mc_bridge,
}


def plot_case_study(show=True, save_path=None):
    if not show:
        import matplotlib
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    params = pd.read_csv(PARAMS_FILE).set_index("parameter")["value"].astype(float)
    paths = simulate_gbm(params.get("S0", 100.0), params.get("r", 0.04), params.get("q", 0.01), params.get("sigma", 0.22), params.get("T", 1.0), int(params.get("steps", 64)), 200, seed=11)
    case_name = "barrier_mc_bridge"
    if case_name == "svi_surface":
        fig = plt.figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        k = np.linspace(-0.35, 0.35, 45)
        tau = np.linspace(0.25, 3.0, 35)
        K_GRID, T_GRID = np.meshgrid(k, tau)
        a = params.get("a", 0.02)
        b = params.get("b", 0.15)
        rho = params.get("rho", -0.35)
        m = params.get("m", 0.0)
        svi_sigma = params.get("svi_sigma", 0.25)
        total_var = a + b * (rho * (K_GRID - m) + np.sqrt((K_GRID - m)**2 + svi_sigma*svi_sigma))
        implied_vol = np.sqrt(total_var / T_GRID)
        ax.plot_surface(K_GRID, T_GRID, implied_vol, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
        ax.set_title("Barrier Options: SVI implied-volatility surface")
        ax.set_xlabel("log-moneyness")
        ax.set_ylabel("expiry")
        ax.set_zlabel("implied vol")
    elif case_name in ("brownian_girsanov", "american_lsm", "barrier_mc_bridge", "asian_mc_control_variate", "mc_variance_reduction_lsm", "merton_jump_diffusion"):
        fig = plt.figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        sample = paths[:55]
        t_grid, p_grid = np.meshgrid(np.arange(sample.shape[1]), np.arange(sample.shape[0]))
        ax.plot_surface(t_grid, p_grid, sample, cmap="plasma", linewidth=0, antialiased=True, alpha=0.88)
        ax.set_title("Barrier Options: simulated path surface")
        ax.set_xlabel("time step")
        ax.set_ylabel("path index")
        ax.set_zlabel("underlying")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(paths[:35].T, alpha=0.45, linewidth=1)
        ax.set_title("Barrier Options: simulated risk-neutral paths")
        ax.set_xlabel("time step")
        ax.set_ylabel("underlying")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    outputs = case_study()
    print("Barrier Options")
    print("-" * len("Barrier Options"))
    for key, value in outputs.items():
        print(f"\n{key}")
        print("-" * len(key))
        if isinstance(value, pd.DataFrame):
            print(value.round(6).to_string(index=False))
        else:
            print(value)
    plot_case_study(show=False, save_path=Path(__file__).with_suffix(".png"))
