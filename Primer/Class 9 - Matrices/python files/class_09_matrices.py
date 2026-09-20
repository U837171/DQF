
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_09_matrices_data.csv"
SCENARIO_FILE = "class_09_matrices_scenarios.csv"
TITLE = "Class 9: Matrices"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def covariance_matrix(data: pd.DataFrame) -> np.ndarray:
    return data[["rates", "equity", "fx"]].cov().to_numpy()


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights.T @ cov @ weights))


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    cov = covariance_matrix(data)
    weights = scenarios["weight"].to_numpy()
    vals, vecs = np.linalg.eigh(cov)
    return {
        "covariance_matrix": cov,
        "portfolio_volatility": portfolio_vol(weights, cov),
        "eigenvalues": vals,
        "first_principal_component": vecs[:, -1],
    }


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    cov = covariance_matrix(data)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cov, cmap="viridis")
    ax.set_xticks(range(3), ["Rates", "Equity", "FX"])
    ax.set_yticks(range(3), ["Rates", "Equity", "FX"])
    ax.set_title("Covariance matrix heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046)
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
