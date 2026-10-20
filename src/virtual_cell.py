"""A four-gene synthetic simulator. All interactions are assumed, not biological facts."""
from __future__ import annotations

import numpy as np

GENES = ["TF_A", "TF_B", "Growth", "Stress"]
SEED = 7204


def production(x: np.ndarray, rates: np.ndarray, doses: tuple[float, float]) -> np.ndarray:
    """Hill-type regulation; knockdown scales production of A or B only."""
    a, b, _, _ = x.T
    activate = lambda z: z**2 / (0.8**2 + z**2)
    return rates * np.column_stack([
        np.full(len(x), 1.4 * (1 - doses[0])),
        (0.2 + 1.6 * activate(a)) * (1 - doses[1]),
        0.2 + 1.8 * activate(b),
        0.2 + 1.5 * (1 - activate(b)),
    ])


def simulate(initial: np.ndarray, rates: np.ndarray,
             doses: tuple[float, float] = (0.0, 0.0), duration: float = 12.0,
             dt: float = 0.02) -> np.ndarray:
    """Vectorized forward Euler integration of dx/dt = production - 0.8*x."""
    x = np.array(initial, dtype=float, copy=True)
    for _ in range(round(duration / dt)):
        x += dt * (production(x, rates, doses) - 0.8 * x)
    return x


def generate(n_cells: int = 400, seed: int = SEED):
    """Paired synthetic cells across conditions, with separate measurement noise."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rates = rng.lognormal(0, 0.22, size=(n_cells, 4))
    baseline = simulate(np.ones_like(rates), rates, duration=40)
    observed_baseline = baseline * rng.lognormal(0, 0.03, size=baseline.shape)
    conditions = [("Control", 0., 0., "standard")]
    for target in ["A", "B"]:
        for dose in [0.25, 0.5, 0.75, 0.6]:
            conditions.append((f"{target}_{dose:g}", dose if target == "A" else 0.,
                               dose if target == "B" else 0.,
                               "novel_dose" if dose == 0.6 else "standard"))
    frames = []
    for name, da, db, domain in conditions:
        endpoint = simulate(baseline, rates, (da, db))
        measured = endpoint * rng.lognormal(0, 0.03, size=endpoint.shape)
        frame = pd.DataFrame({"cell_id": np.arange(n_cells), "condition": name,
                              "dose_A": da, "dose_B": db, "domain": domain})
        for i, gene in enumerate(GENES):
            frame[f"baseline_{gene}"] = observed_baseline[:, i]
            frame[f"post_{gene}"] = measured[:, i]
            frame[f"lfc_{gene}"] = np.log2(measured[:, i] / observed_baseline[:, i])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
