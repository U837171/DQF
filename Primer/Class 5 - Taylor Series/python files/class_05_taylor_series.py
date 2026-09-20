
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_05_taylor_series_data.csv"
SCENARIO_FILE = "class_05_taylor_series_scenarios.csv"
TITLE = "Class 5: Taylor Series"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def exact_value(spot: np.ndarray) -> np.ndarray:
    return 0.004 * spot ** 2 + 1.5 * np.log(spot)


def taylor_second_order(s0: float, ds: np.ndarray) -> np.ndarray:
    delta = 0.008 * s0 + 1.5 / s0
    gamma = 0.008 - 1.5 / (s0 ** 2)
    return exact_value(np.array([s0]))[0] + delta * ds + 0.5 * gamma * ds ** 2


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    s0 = float(data["spot"].iloc[0])
    ds = scenarios["shock"].to_numpy()
    out = scenarios[["scenario", "shock"]].copy()
    out["exact_change"] = exact_value(s0 + ds) - exact_value(np.array([s0]))[0]
    out["taylor_change"] = taylor_second_order(s0, ds) - exact_value(np.array([s0]))[0]
    out["error"] = out["taylor_change"] - out["exact_change"]
    return {"taylor_pnl_table": out, "s0": s0, "max_abs_error": float(out["error"].abs().max())}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    s0 = float(data["spot"].iloc[0])
    ds = np.linspace(scenarios["shock"].min(), scenarios["shock"].max(), 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s0 + ds, exact_value(s0 + ds) - exact_value(np.array([s0]))[0], label="Exact change", lw=2)
    ax.plot(s0 + ds, taylor_second_order(s0, ds) - exact_value(np.array([s0]))[0], "--", label="Second-order Taylor")
    ax.set_title("Taylor approximation of nonlinear PnL")
    ax.set_xlabel("Spot after shock")
    ax.set_ylabel("Change in value")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def run_case_study(save_plots: bool = True) -> Dict[str, object]:
    """Run the main case study and return reusable numerical outputs."""
    data = load_data()
    scenarios = load_scenarios()
    outputs = analyze(data, scenarios)
    print(f"\n{TITLE}")
    print("-" * len(TITLE))
    for key, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n{key}:")
            print(value.round(6).to_string(index=False))
        elif isinstance(value, pd.Series):
            print(f"\n{key}:")
            numeric = pd.to_numeric(value, errors="coerce")
            formatted = value.copy()
            for idx, number in numeric.dropna().items():
                formatted.loc[idx] = round(float(number), 6)
            print(formatted.to_string())
        elif isinstance(value, np.ndarray):
            print(f"\n{key}:\n{np.round(value, 6)}")
        else:
            print(f"{key}: {value}")
    if save_plots:
        try:
            plot_case_study(data, scenarios, Path(__file__).with_suffix(".png"), show=False)
            print(f"\nSaved plot: {Path(__file__).with_suffix('.png').name}")
        except Exception as exc:
            print(f"\nPlot skipped: {exc}")
    return outputs


if __name__ == "__main__":
    run_case_study(save_plots=True)
