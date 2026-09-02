
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_19_named_distributions_data.csv"
SCENARIO_FILE = "class_19_named_distributions_scenarios.csv"
TITLE = "Class 19: Named Distributions"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def normal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)((x - mu) / (sigma * math.sqrt(2.0))))


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    alpha = float(scenarios["alpha"].iloc[0])
    losses = data["normal_loss"].to_numpy()
    var = float(np.quantile(losses, alpha))
    lognormal_mean = float(data["lognormal_price"].mean())
    exp_mean = float(data["exponential_wait"].mean())
    return {"normal_var": var, "lognormal_price_mean": lognormal_mean, "exponential_wait_mean": exp_mean, "sample_summary": data.describe()}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data["normal_loss"], bins=30, alpha=0.55, label="Normal loss", color="#2f6f9f")
    ax.hist(data["lognormal_price"], bins=30, alpha=0.45, label="Lognormal price", color="#de8f05")
    ax.set_title("Named distributions used in market-risk examples")
    ax.legend()
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
