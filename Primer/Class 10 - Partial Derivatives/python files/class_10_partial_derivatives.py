
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_10_partial_derivatives_data.csv"
SCENARIO_FILE = "class_10_partial_derivatives_scenarios.csv"
TITLE = "Class 10: Partial Derivatives"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def value_surface(s: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return 0.04 * s * sigma + 0.002 * s ** 2 + 3.0 * sigma ** 2


def gradient_and_hessian(s: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    grad = np.array([0.04 * sigma + 0.004 * s, 0.04 * s + 6.0 * sigma])
    hess = np.array([[0.004, 0.04], [0.04, 6.0]])
    return grad, hess


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    s0 = float(data["spot"].iloc[0])
    vol0 = float(data["vol"].iloc[0])
    grad, hess = gradient_and_hessian(s0, vol0)
    dx = scenarios[["dS", "dvol"]].to_numpy()
    pnl = dx @ grad + 0.5 * np.einsum("ij,jk,ik->i", dx, hess, dx)
    out = scenarios.copy()
    out["second_order_pnl"] = pnl
    return {"gradient": grad, "hessian": hess, "scenario_pnl": out}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    s = np.linspace(80, 120, 40)
    v = np.linspace(0.1, 0.5, 40)
    S, V = np.meshgrid(s, v)
    Z = value_surface(S, V)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S, V, Z, cmap="viridis", alpha=0.9, linewidth=0)
    ax.set_title("Valuation surface with spot and volatility")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Volatility")
    ax.set_zlabel("Value")
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
