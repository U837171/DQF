
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_16_probability_theory_ii_data.csv"
SCENARIO_FILE = "class_16_probability_theory_ii_scenarios.csv"
TITLE = "Class 16: Probability Theory II"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def weighted_moments(losses: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    mean = float(np.sum(probs * losses))
    var = float(np.sum(probs * (losses - mean) ** 2))
    return mean, var


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    mean, var = weighted_moments(data["loss"].to_numpy(), data["probability"].to_numpy())
    transformed = np.exp(data["return"].to_numpy()) - 1
    return {
        "expected_loss": mean,
        "loss_variance": var,
        "transformed_return_mean": float(transformed.mean()),
        "scenario_table": data,
    }


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(data["scenario"], data["loss"], color="#2f6f9f", alpha=0.8)
    ax.set_title("Discrete scenario losses for expectation")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Loss")
    ax.grid(axis="y", alpha=0.25)
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
