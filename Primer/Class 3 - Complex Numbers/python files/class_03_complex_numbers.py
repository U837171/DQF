
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_FILE = "class_03_complex_numbers_data.csv"
SCENARIO_FILE = "class_03_complex_numbers_scenarios.csv"
TITLE = "Class 3: Complex Numbers"


def load_data() -> pd.DataFrame:
    """Load the main topic data from the local folder."""
    return pd.read_csv(DATA_FILE)


def load_scenarios() -> pd.DataFrame:
    """Load the scenario table from the local folder."""
    return pd.read_csv(SCENARIO_FILE)


def complex_return(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return amplitude * (np.cos(phase) + 1j * np.sin(phase))


def analyze(data: pd.DataFrame, scenarios: pd.DataFrame) -> Dict[str, object]:
    z = complex_return(data["amplitude"].to_numpy(), data["phase"].to_numpy())
    table = data[["trade_id", "amplitude", "phase"]].copy()
    table["real"] = z.real
    table["imaginary"] = z.imag
    table["modulus"] = np.abs(z)
    table["argument"] = np.angle(z)
    return {"complex_factor_table": table, "mean_modulus": float(table["modulus"].mean())}


def plot_case_study(data: pd.DataFrame, scenarios: pd.DataFrame, save_path: Path | None = None, show: bool = True):
    import matplotlib.pyplot as plt
    z = complex_return(data["amplitude"].to_numpy(), data["phase"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z.real, z.imag, s=80, c=data["amplitude"], cmap="viridis", edgecolor="black")
    for trade, x, y in zip(data["trade_id"], z.real, z.imag):
        ax.annotate(trade, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Complex factor shocks in the plane")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
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
