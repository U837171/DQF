
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_08_vectors_data.csv"
SCENARIO_FILE = "class_08_vectors_scenarios.csv"
TITLE = "Class 8: Vectors"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def portfolio_delta(exposures: pd.DataFrame, shocks: pd.Series) -> pd.Series:
    return exposures.set_index("trade_id")[["rates", "equity", "fx"]].dot(shocks)


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    shock = scenarios.set_index("factor")["shock"]
    pnl = portfolio_delta(data, shock)
    vector = data[["rates", "equity", "fx"]].sum().to_numpy()
    hedge = np.array([1.0, -0.3, 0.2])
    projection = hedge * (np.dot(vector, hedge) / np.dot(hedge, hedge))
    return {
        "trade_linear_pnl": pnl.reset_index(name="linear_pnl"),
        "portfolio_exposure_vector": vector,
        "l2_norm": float(np.linalg.norm(vector)),
        "projection_on_candidate_hedge": projection,
    }


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    vec = data[["rates", "equity", "fx"]].sum().to_numpy()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color="#2f6f9f", linewidth=3)
    ax.set_title("3D portfolio exposure vector")
    ax.set_xlabel("Rates")
    ax.set_ylabel("Equity")
    ax.set_zlabel("FX")
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
