
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_13_numerical_optimization_data.csv"
SCENARIO_FILE = "class_13_numerical_optimization_scenarios.csv"
TITLE = "Class 13: Numerical Optimization"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def f(x: float) -> float:
    return (x - 2.0) ** 4 + 0.4 * (x + 1.0) ** 2


def fp(x: float) -> float:
    return 4.0 * (x - 2.0) ** 3 + 0.8 * (x + 1.0)


def fpp(x: float) -> float:
    return 12.0 * (x - 2.0) ** 2 + 0.8


def newton_path(x0: float, steps: int = 8) -> pd.DataFrame:
    x = float(x0)
    rows = []
    for k in range(steps):
        rows.append({"step": k, "x": x, "objective": f(x), "gradient": fp(x)})
        x = x - fp(x) / fpp(x)
    return pd.DataFrame(rows)


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    path = newton_path(float(data["initial_x"].iloc[0]), int(data["steps"].iloc[0]))
    return {"newton_iterations": path, "final_x": float(path["x"].iloc[-1])}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    xgrid = np.linspace(-2.5, 4.0, 300)
    path = newton_path(float(data["initial_x"].iloc[0]), int(data["steps"].iloc[0]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xgrid, [f(x) for x in xgrid], color="#2f6f9f")
    ax.scatter(path["x"], path["objective"], color="#de8f05", zorder=3)
    ax.set_title("Newton iterations on a convex objective")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Objective")
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
