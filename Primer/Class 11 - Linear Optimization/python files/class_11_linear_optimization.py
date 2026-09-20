
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_11_linear_optimization_data.csv"
SCENARIO_FILE = "class_11_linear_optimization_scenarios.csv"
TITLE = "Class 11: Linear Optimization"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def enumerate_vertices(constraints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    A = constraints[["a1", "a2"]].to_numpy()
    b = constraints["b"].to_numpy()
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            M = np.vstack([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-10:
                continue
            x = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ x <= b + 1e-9):
                rows.append({"x1": x[0], "x2": x[1], "binding": f"{i},{j}"})
    return pd.DataFrame(rows).drop_duplicates(subset=["x1", "x2"])


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    vertices = enumerate_vertices(data)
    c = scenarios["cost"].to_numpy()
    vertices["objective_cost"] = vertices[["x1", "x2"]].to_numpy() @ c
    best = vertices.loc[vertices["objective_cost"].idxmin()]
    return {"feasible_vertices": vertices, "minimum_cost_solution": best}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    verts = enumerate_vertices(data)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(verts["x1"], verts["x2"], s=80, color="#de8f05")
    for _, r in data.iterrows():
        x = np.linspace(0, max(verts["x1"].max() * 1.2, 1), 100)
        if abs(r["a2"]) > 1e-12:
            y = (r["b"] - r["a1"] * x) / r["a2"]
            ax.plot(x, y, alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Feasible hedge region from linear constraints")
    ax.set_xlabel("Hedge 1 notional")
    ax.set_ylabel("Hedge 2 notional")
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
