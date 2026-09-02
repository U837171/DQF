
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_18_conditional_expectation_data.csv"
SCENARIO_FILE = "class_18_conditional_expectation_scenarios.csv"
TITLE = "Class 18: Conditional Expectation"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def conditional_mean_by_bucket(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["equity_bucket"] = pd.cut(out["equity_return"], bins=[-1, -0.02, 0.0, 0.02, 1], labels=["selloff", "down", "up", "rally"])
    return out.groupby("equity_bucket", observed=True)["loss"].agg(["count", "mean", "std"]).reset_index()


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    bucket_means = conditional_mean_by_bucket(data)
    unconditional = float(data["loss"].mean())
    tower_check = float((bucket_means["count"] * bucket_means["mean"]).sum() / bucket_means["count"].sum())
    return {"conditional_loss_means": bucket_means, "unconditional_mean": unconditional, "tower_property_check": tower_check}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    means = conditional_mean_by_bucket(data)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(means["equity_bucket"].astype(str), means["mean"], color="#2f6f9f", alpha=0.85)
    ax.set_title("Conditional expected loss by equity-return bucket")
    ax.set_xlabel("Equity-return bucket")
    ax.set_ylabel("Mean loss")
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
