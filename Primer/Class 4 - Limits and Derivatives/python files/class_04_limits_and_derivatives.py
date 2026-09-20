
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_04_limits_and_derivatives_data.csv"
SCENARIO_FILE = "class_04_limits_and_derivatives_scenarios.csv"
TITLE = "Class 4: Limits and Derivatives"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def value_curve(s: np.ndarray) -> np.ndarray:
    return 0.015 * s ** 2 - 2.0 * np.log(s)


def finite_difference_delta(s: float, h: float = 0.01) -> float:
    central_difference = (value_curve(np.array([s + h])) - value_curve(np.array([s - h]))) / (2 * h)
    return float(central_difference[0])


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    out = data[["spot"]].copy()
    out["value"] = value_curve(out["spot"].to_numpy())
    out["finite_difference_delta"] = [finite_difference_delta(float(s), 0.01) for s in out["spot"]]
    out["analytic_delta"] = 0.03 * out["spot"] - 2.0 / out["spot"]
    out["absolute_error"] = (out["finite_difference_delta"] - out["analytic_delta"]).abs()
    return {"delta_table": out, "max_delta_error": float(out["absolute_error"].max())}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    s = np.linspace(data["spot"].min() * 0.8, data["spot"].max() * 1.2, 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s, value_curve(s), color="#2f6f9f", label="Value curve")
    ax.scatter(data["spot"], value_curve(data["spot"].to_numpy()), color="#de8f05", zorder=3)
    ax.set_title("Limit slope as local risk sensitivity")
    ax.set_xlabel("Underlying level")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend()
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
