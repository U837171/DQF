
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_07_differential_equations_data.csv"
SCENARIO_FILE = "class_07_differential_equations_scenarios.csv"
TITLE = "Class 7: Differential Equations"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def euler_mean_reversion(x0: float, kappa: float, theta: float, dt: float, n: int) -> np.ndarray:
    path = np.empty(n + 1)
    path[0] = x0
    for i in range(n):
        path[i + 1] = path[i] + kappa * (theta - path[i]) * dt
    return path


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    row = data.iloc[0]
    path = euler_mean_reversion(row["x0"], row["kappa"], row["theta"], row["dt"], int(row["steps"]))
    table = pd.DataFrame({"step": np.arange(len(path)), "rate": path})
    table["discount_factor"] = np.exp(-table["rate"].cumsum() * row["dt"])
    return {"mean_reversion_path": table.head(12), "terminal_rate": float(path[-1])}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    row = data.iloc[0]
    path = euler_mean_reversion(row["x0"], row["kappa"], row["theta"], row["dt"], int(row["steps"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(path)) * row["dt"], path, color="#2f6f9f", lw=2)
    ax.axhline(row["theta"], color="#de8f05", ls="--", label="Long-run mean")
    ax.set_title("Euler solution of a mean-reverting short-rate model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate")
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
