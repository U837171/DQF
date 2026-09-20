
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_06_integration_data.csv"
SCENARIO_FILE = "class_06_integration_scenarios.csv"
TITLE = "Class 6: Integration"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def discount_factor(rate: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.exp(-rate * t)


def trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    data = data.copy()
    data["df"] = discount_factor(data["zero_rate"], data["time"])
    data["discounted_exposure"] = data["expected_exposure"] * data["df"]
    area = trapezoid_area(data["time"].to_numpy(), data["discounted_exposure"].to_numpy())
    return {"discounted_exposure_table": data, "exposure_pv_area": area}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    df = data.copy()
    df["discounted_exposure"] = df["expected_exposure"] * np.exp(-df["zero_rate"] * df["time"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(df["time"], df["discounted_exposure"], alpha=0.35, color="#2f6f9f")
    ax.plot(df["time"], df["discounted_exposure"], color="#2f6f9f", lw=2)
    ax.set_title("Definite integral as area under discounted exposure")
    ax.set_xlabel("Time")
    ax.set_ylabel("Discounted expected exposure")
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
