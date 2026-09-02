
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_17_probability_theory_iii_data.csv"
SCENARIO_FILE = "class_17_probability_theory_iii_scenarios.csv"
TITLE = "Class 17: Probability Theory III"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def var_es(losses: np.ndarray, alpha: float) -> Tuple[float, float]:
    var = float(np.quantile(losses, alpha))
    es = float(losses[losses >= var].mean())
    return var, es


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    cov = data[["rates", "equity"]].cov()
    corr = data[["rates", "equity"]].corr()
    pnl = 1000 * data["rates"] + 700 * data["equity"]
    losses = -pnl.to_numpy()
    var, es = var_es(losses, float(scenarios["alpha"].iloc[0]))
    return {"covariance": cov, "correlation": corr, "portfolio_var": var, "portfolio_es": es}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data["rates"], data["equity"], alpha=0.65, color="#2f6f9f")
    ax.set_title("Joint risk-factor shocks and correlation")
    ax.set_xlabel("Rates shock")
    ax.set_ylabel("Equity shock")
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
