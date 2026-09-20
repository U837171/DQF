
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_14_numerical_integration_data.csv"
SCENARIO_FILE = "class_14_numerical_integration_scenarios.csv"
TITLE = "Class 14: Numerical Integration"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def exposure_function(t: np.ndarray) -> np.ndarray:
    return 2.0 + 0.6 * t + 0.8 * np.exp(-0.7 * t)


def trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def simpson(x: np.ndarray, y: np.ndarray) -> float:
    if (len(x) - 1) % 2 != 0:
        raise ValueError("Simpson requires an even number of intervals.")
    h = (x[-1] - x[0]) / (len(x) - 1)
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum()))


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    x = data["time"].to_numpy()
    y = data["exposure"].to_numpy()
    return {"trapezoid_estimate": trapezoid(x, y), "simpson_estimate": simpson(x, y), "grid": data}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["time"], data["exposure"], color="#2f6f9f", marker="o")
    ax.fill_between(data["time"], data["exposure"], alpha=0.25, color="#2f6f9f")
    ax.set_title("Numerical integration of expected exposure")
    ax.set_xlabel("Time")
    ax.set_ylabel("Exposure")
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
