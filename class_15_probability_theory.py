
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_15_probability_theory_data.csv"
SCENARIO_FILE = "class_15_probability_theory_scenarios.csv"
TITLE = "Class 15: Probability Theory"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def empirical_var(losses: np.ndarray, alpha: float) -> float:
    return float(np.quantile(losses, alpha))


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    losses = data["loss"].to_numpy()
    threshold = float(scenarios["threshold"].iloc[0])
    alpha = float(scenarios["alpha"].iloc[0])
    return {
        "probability_loss_exceeds_threshold": float(np.mean(losses > threshold)),
        "empirical_var": empirical_var(losses, alpha),
        "loss_summary": data["loss"].describe(),
    }


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    alpha = float(scenarios["alpha"].iloc[0])
    var = empirical_var(data["loss"].to_numpy(), alpha)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data["loss"], bins=30, color="#2f6f9f", alpha=0.75)
    ax.axvline(var, color="#de8f05", lw=2, label=f"VaR {alpha:.0%}")
    ax.set_title("Loss distribution and quantile event")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Frequency")
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
