
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_12_non_linear_optimization_data.csv"
SCENARIO_FILE = "class_12_non_linear_optimization_scenarios.csv"
TITLE = "Class 12: Non Linear Optimization"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    a, b = theta
    residual = y - (a + b * np.exp(-x))
    return float(np.sum(residual ** 2))


def gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = theta
    residual = y - (a + b * np.exp(-x))
    return np.array([-2 * residual.sum(), -2 * np.sum(residual * np.exp(-x))])


def gradient_descent(x: np.ndarray, y: np.ndarray, theta0: np.ndarray, lr: float = 0.05, steps: int = 200) -> pd.DataFrame:
    theta = theta0.astype(float).copy()
    rows = []
    for k in range(steps):
        rows.append({"step": k, "a": theta[0], "b": theta[1], "objective": objective(theta, x, y)})
        theta -= lr * gradient(theta, x, y)
    return pd.DataFrame(rows)


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    path = gradient_descent(data["tenor"].to_numpy(), data["zero_rate"].to_numpy(), np.array([0.01, 0.04]), 0.02, 160)
    return {"calibration_path_tail": path.tail(8), "final_parameters": path.iloc[-1][["a", "b"]]}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    x = data["tenor"].to_numpy()
    y = data["zero_rate"].to_numpy()
    path = gradient_descent(x, y, np.array([0.01, 0.04]), 0.02, 160)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    A, B = np.meshgrid(np.linspace(0.0, 0.06, 45), np.linspace(-0.02, 0.08, 45))
    Z = np.vectorize(lambda a, b: objective(np.array([a, b]), x, y))(A, B)
    ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.85, linewidth=0)
    ax.plot(path["a"], path["b"], path["objective"], color="white", lw=2)
    ax.set_title("Nonlinear calibration objective surface")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("SSE")
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
