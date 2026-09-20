
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_01_set_theory_basics_data.csv"
SCENARIO_FILE = "class_01_set_theory_basics_scenarios.csv"
TITLE = "Class 1: Set Theory Basics"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def exposure_sets(data: pd.DataFrame) -> Dict[str, set]:
    return {col: set(data.loc[data[col] == 1, "trade_id"]) for col in ["rates", "equity", "fx", "credit"]}


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    sets = exposure_sets(data)
    rates_or_equity = sets["rates"] | sets["equity"]
    both_rates_equity = sets["rates"] & sets["equity"]
    all_market = sets["rates"] | sets["equity"] | sets["fx"] | sets["credit"]
    pnl_by_scenario = scenarios.merge(data, on="trade_id")
    pnl_by_scenario["pnl"] = (
        pnl_by_scenario["delta"] * pnl_by_scenario["shock"]
        + 0.5 * pnl_by_scenario["gamma"] * pnl_by_scenario["shock"] ** 2
    )
    return {
        "number_of_trades": len(data),
        "rates_or_equity_count": len(rates_or_equity),
        "rates_and_equity_count": len(both_rates_equity),
        "market_risk_universe_count": len(all_market),
        "scenario_pnl_by_book": pnl_by_scenario.groupby("book", as_index=False)["pnl"].sum(),
    }


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    counts = data[["rates", "equity", "fx", "credit"]].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", ax=ax, color=["#2f6f9f", "#de8f05", "#029e73", "#cc78bc"])
    ax.set_title("Risk-factor membership by set")
    ax.set_ylabel("Trade count")
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
