
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_02_relations_and_functions_data.csv"
SCENARIO_FILE = "class_02_relations_and_functions_scenarios.csv"
TITLE = "Class 2: Relations and Functions"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def forward_value(spot: np.ndarray, rate: np.ndarray, maturity: np.ndarray, strike: np.ndarray) -> np.ndarray:
    return spot - strike * np.exp(-rate * maturity)


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    values = forward_value(data["spot"], data["rate"], data["maturity"], data["strike"])
    bumped = forward_value(data["spot"] + 1.0, data["rate"], data["maturity"], data["strike"])
    result = data[["trade_id", "spot", "strike", "rate", "maturity"]].copy()
    result["value"] = values
    result["spot_bump_delta"] = bumped - values
    result["risk_bucket"] = pd.cut(result["value"], bins=[-100, -2, 2, 100], labels=["negative", "near_zero", "positive"])
    return {"function_outputs": result, "average_delta": float(result["spot_bump_delta"].mean())}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    spots = np.linspace(data["spot"].min() * 0.9, data["spot"].max() * 1.1, 80)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _, row in data.head(4).iterrows():
        ax.plot(spots, forward_value(spots, row["rate"], row["maturity"], row["strike"]), label=row["trade_id"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Forward value as a function of spot")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Value")
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
