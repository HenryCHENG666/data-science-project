"""Run from anywhere: python src/analyze.py. Writes all data, metrics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from virtual_cell import GENES, SEED, generate

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0e938b", "#5865c8", "#f59e42", "#df637e"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#31435b", "text.color": "#172b46",
                     "axes.titleweight": "bold", "figure.facecolor": "#fbfcff",
                     "axes.facecolor": "#fbfcff", "savefig.facecolor": "#fbfcff",
                     "svg.fonttype": "none"})


def save(fig, name):
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def network():
    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.set(xlim=(-.75, 8.5), ylim=(-1.25, 1.4))
    ax.axis("off")
    positions = [(0, 0), (2.4, 0), (5.2, .65), (5.2, -.65)]
    for (x, y), gene, c in zip(positions, GENES, COLORS):
        ax.add_patch(FancyBboxPatch((x-.48, y-.27), 1.45, .54,
                     boxstyle="round,pad=.12", linewidth=0, facecolor=c))
        ax.text(x+.245, y, gene, color="white", ha="center", va="center", weight="bold")
    for start, end, label in [((1.05, 0), (1.8, 0), "+"),
                               ((3.45, .08), (4.6, .55), "+"),
                               ((3.45, -.08), (4.6, -.55), "−")]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=18, color="#61738d", lw=2))
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.15, label, fontsize=15)
    ax.text(7.05, .6, "INPUT", fontsize=10, color="#61738d", weight="bold")
    ax.text(7.05, .25, "A or B inhibition\n25%, 50%, 75%", fontsize=11)
    ax.text(7.05, -.55, "New-dose test: 60%", fontsize=10, color=COLORS[0])
    ax.set_title("A small, transparent virtual-cell sandbox", loc="left", pad=17)
    fig.text(.13, .03, "Assumed interactions • arbitrary expression units • synthetic data only", color="#61738d", fontsize=10)
    save(fig, "01_network")


def main():
    for folder in ["data", "results", "figures"]:
        (ROOT / folder).mkdir(exist_ok=True)
    df = generate()
    df.to_csv(ROOT / "data" / "synthetic_cells.csv", index=False, float_format="%.8f")
    standard = df[df.domain == "standard"].reset_index(drop=True)
    features = [f"baseline_{g}" for g in GENES] + ["dose_A", "dose_B"]
    targets = [f"lfc_{g}" for g in GENES]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
                               .split(standard, groups=standard.cell_id))
    train, test = standard.iloc[train_idx], standard.iloc[test_idx]
    test_ids = set(test.cell_id)
    assert not (set(train.cell_id) & test_ids), "Cell leakage"
    novel = df[(df.domain == "novel_dose") & df.cell_id.isin(test_ids)]
    models = {"Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
              "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                     random_state=SEED, n_jobs=1)}
    for model in models.values():
        model.fit(train[features], train[targets])
    metrics, predictions = [], []
    for domain, frame in [("Held-out cells", test), ("New dose (60%)", novel)]:
        truth = frame[targets].to_numpy()
        for name in ["No response", *models]:
            pred = np.zeros_like(truth) if name == "No response" else models[name].predict(frame[features])
            metrics.append({"evaluation": domain, "model": name,
                            "MAE": mean_absolute_error(truth, pred),
                            "R2": r2_score(truth, pred)})
            rows = frame[["cell_id", "condition"]].copy()
            rows["evaluation"], rows["model"] = domain, name
            for i, g in enumerate(GENES):
                rows[f"true_{g}"], rows[f"pred_{g}"] = truth[:, i], pred[:, i]
            predictions.append(rows)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(ROOT / "results" / "metrics.csv", index=False)
    pd.concat(predictions).to_csv(ROOT / "results" / "predictions.csv", index=False, float_format="%.8f")
    split = {"seed": SEED, "train_cell_ids": sorted(map(int, train.cell_id.unique())),
             "test_cell_ids": sorted(map(int, test.cell_id.unique())),
             "training_rows": len(train), "held_out_rows": len(test), "new_dose_rows": len(novel)}
    (ROOT / "results" / "split.json").write_text(json.dumps(split, indent=2)+"\n")
    effect = test.groupby("condition")[targets].mean()
    effect.columns = GENES
    order = ["Control", "A_0.25", "A_0.5", "A_0.75", "B_0.25", "B_0.5", "B_0.75"]
    effect = effect.loc[order]
    effect.to_csv(ROOT / "results" / "mean_effects.csv")
    network()

    fig, ax = plt.subplots(figsize=(9, 4.7))
    scale = np.abs(effect.to_numpy()).max()
    im = ax.imshow(effect.T, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    ax.set_xticks(range(7), ["Control", "A 25%", "A 50%", "A 75%", "B 25%", "B 50%", "B 75%"])
    ax.set_yticks(range(4), GENES)
    for i in range(4):
        for j in range(7):
            value = effect.iloc[j, i]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > scale*.58 else "#172b46")
    ax.set_title("Gene inhibition propagates through the assumed network", loc="left", pad=20)
    fig.colorbar(im, ax=ax, label="Mean log₂ fold change", shrink=.85)
    fig.tight_layout()
    save(fig, "02_response_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, domain in zip(axes, metrics.evaluation.unique()):
        subset = metrics[metrics.evaluation == domain]
        bars = ax.bar(subset.model, subset.MAE, color=["#a4afbf", COLORS[1], COLORS[0]], width=.62)
        ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
        ax.set_title(domain, loc="left", pad=18)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, metrics.MAE.max()*1.24)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean absolute error (log₂ fold change) ↓")
    fig.suptitle("Prediction quality: familiar doses vs an unseen dose", x=.08, ha="left", weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "03_model_comparison")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    pred = models["Random forest"].predict(novel[features])
    for i, (ax, g, c) in enumerate(zip(axes, GENES, COLORS)):
        true = novel[f"lfc_{g}"].to_numpy()
        ax.scatter(true, pred[:, i], s=16, alpha=.55, color=c, edgecolors="none")
        lo, hi = min(true.min(), pred[:, i].min())-.05, max(true.max(), pred[:, i].max())+.05
        ax.plot([lo, hi], [lo, hi], "--", color="#61738d", lw=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=g, xlabel="Simulated response")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Random forest prediction")
    fig.suptitle("Unseen 60% inhibition: errors remain visible", x=.07, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "04_prediction_scatter")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)
    dose_rows = []
    for ax, target in zip(axes, ["A", "B"]):
        sub = df[((df.dose_A > 0) if target == "A" else (df.dose_B > 0)) & df.cell_id.isin(test_ids)]
        for g, c in zip(GENES, COLORS):
            grouped = sub.groupby(f"dose_{target}")[f"lfc_{g}"]
            means, ses = grouped.mean(), grouped.std()/np.sqrt(grouped.count())
            ax.errorbar(means.index*100, means, yerr=1.96*ses, color=c, marker="o", capsize=3, label=g)
            dose_rows += [{"target": target, "gene": g, "dose": d, "mean_lfc": float(means[d]),
                           "ci95_halfwidth": float(1.96*ses[d])} for d in means.index]
        ax.axvline(60, ls=":", color="#8b98ab", lw=1)
        ax.axhline(0, color="#8b98ab", lw=.6)
        ax.set_title(f"Inhibit TF_{target}", loc="left")
        ax.set_xlabel("Production inhibition (%)")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("Mean log₂ fold change (± 1.96 SE)")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Dose response follows the simulator's designed interactions", x=.08, ha="left", weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "05_dose_response")
    pd.DataFrame(dose_rows).to_csv(ROOT / "results" / "dose_response.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    importances = models["Random forest"].feature_importances_
    idx = np.argsort(importances)
    names = [s.replace("baseline_", "Initial ").replace("dose_", "Inhibit ") for s in features]
    ax.barh(np.array(names)[idx], importances[idx], color=[COLORS[0] if i >= 4 else "#9babc3" for i in idx])
    ax.set_xlabel("Impurity-based importance (predictive, not causal)")
    ax.set_title("Perturbation inputs carry most predictive information", loc="left", pad=18)
    ax.grid(axis="x", alpha=.12)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "06_feature_importance")
    pd.DataFrame({"feature": features, "importance": importances}).to_csv(ROOT / "results" / "feature_importance.csv", index=False)
    summary = {"synthetic": True, "cells": int(df.cell_id.nunique()), "rows": len(df),
               "genes": GENES, "metrics": metrics.to_dict(orient="records"),
               "mean_effects": effect.round(4).to_dict(orient="index")}
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(metrics.to_string(index=False))
    print(f"Saved {len(df)} rows, metrics, split manifest and six figures in the project directory.")


if __name__ == "__main__":
    main()
