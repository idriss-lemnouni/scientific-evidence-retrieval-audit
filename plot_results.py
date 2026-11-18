"""Render publication figures from verified run artifacts."""
from collections import defaultdict
from math import isclose
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK = "#172b3b"
MUTED = "#526575"
TEAL = "#007e78"
BLUE = "#4069ba"
ORANGE = "#b9502d"
SOFT = "#d5e1e7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": SOFT, "axes.axisbelow": True, "grid.color": SOFT,
    "grid.linewidth": 0.7, "legend.frameon": False,
})


def save(fig, out, name, title, note):
    fig.suptitle(title, x=0.055, y=0.96, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.055, 0.035, note, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0.025, .085, .995, .87), pad=1.5)
    fig.savefig(out / (name + ".png"), dpi=180, facecolor="white")
    plt.close(fig)

def render(run, out):
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    scores = report["results"]["test"]
    size = report["provenance"]["test_queries"]
    models = [("bm25", "Weighted BM25", TEAL), ("phrase-fusion", "BM25 + phrase fusion", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.25), sharey=True)
    ranks = [1, 5, 10]
    for axis, (key, name) in zip(axes, [("ndcg", "nDCG"), ("recall", "Recall"), ("mrr", "MRR")]):
        for model, label, color in models:
            axis.plot(ranks, [scores[model][f"{key}@{k}"] for k in ranks], "o-",
                      color=color, label=label, linewidth=2, markersize=5)
        axis.set(title=name, xlabel="Rank cutoff", xticks=ranks, ylim=(0, 1))
        axis.grid(axis="y")
    axes[0].set_ylabel("Mean score")
    axes[-1].legend(loc="lower right", fontsize=8)
    save(fig, out, "ranking-by-depth", "Weighted BM25 leads across ranking depths",
         f"SciFact · {size} official test queries · averages from paired judgments at ranks 1, 5 and 10")

    paired = defaultdict(dict)
    with (run / "per_query.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test":
                paired[row["queryid"]][row["model"]] = float(row["ndcg@10"])
    if len(paired) != size or any(set(v) != {"bm25", "phrase-fusion"} for v in paired.values()):
        raise ValueError("The held-out results are not paired at the official query count")
    deltas = [v["bm25"] - v["phrase-fusion"] for v in paired.values()]
    counts = [sum(d > 1e-9 for d in deltas), sum(isclose(d, 0, abs_tol=1e-9) for d in deltas),
              sum(d < -1e-9 for d in deltas)]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(["BM25 higher", "Tie", "Phrase fusion higher"], counts, color=[TEAL, "#8496a5", BLUE], width=.62)
    ax.bar_label(bars, padding=4, fontweight="bold")
    ax.set(ylabel="Test queries", ylim=(0, max(counts) * 1.18))
    ax.grid(axis="y")
    save(fig, out, "paired-query-outcomes", "Performance varies across individual queries",
         "Paired nDCG@10 for each held-out query · differences within 10⁻⁹ are treated as ties")

    gate = scores["citation_gate"]
    counts = [gate["matched"], gate["emitted"] - gate["matched"], size - gate["emitted"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(["Citation judged\nrelevant", "Citation without a\npositive judgment", "Abstained"],
                  counts, color=[TEAL, ORANGE, "#8496a5"], width=.62)
    ax.bar_label(bars, padding=4, fontweight="bold")
    ax.set(ylabel="Test queries", ylim=(0, max(counts) * 1.18))
    ax.grid(axis="y")
    save(fig, out, "citation-gate", f"The citation gate emits a candidate for {gate['emitted']} queries",
         f"BM25 · overlap threshold {report['threshold']:.2f} · {size} test queries · relevance is judged at document level")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("figures/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    render(args.run, args.out)
