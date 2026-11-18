"""Score retrieval on judged SciFact queries and calibrate citation coverage."""
from contextlib import closing
import csv
import json
from math import log2
from pathlib import Path
from statistics import mean

from corpus import open_index
from retrieval import MODELS, evidence, search

THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)
DEPTHS = (1, 5, 10)


def metrics(ids: list[str], judged: dict[str, int], limit: int) -> dict:
    returned = ids[:limit]
    grades = [judged.get(doc, 0) for doc in returned]
    ideal = sorted(judged.values(), reverse=True)[:limit]
    dcg = sum((2 ** grade - 1) / log2(rank + 1) for rank, grade in enumerate(grades, 1))
    idcg = sum((2 ** grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    relevant = sum(grade > 0 for grade in judged.values())
    return {
        "ndcg": dcg / idcg if idcg else 0.0,
        "recall": sum(grade > 0 for grade in grades) / relevant if relevant else 0.0,
        "mrr": next((1 / rank for rank, grade in enumerate(grades, 1) if grade > 0), 0.0),
    }


def assess(db, split: str) -> list[dict]:
    queries = {}
    for queryid, text, docid, grade in db.execute("""
        SELECT q.queryid, q.text, j.docid, j.score
        FROM qrels j JOIN queries q USING (queryid)
        WHERE j.split = ? ORDER BY q.queryid, j.docid
    """, (split,)):
        queries.setdefault(queryid, (text, {}))[1][docid] = grade
    results = []
    for queryid, (text, judged) in queries.items():
        for model, hits in search(db, text).items():
            identities = [hit.identity for hit in hits]
            top = hits[0] if hits else None
            score, passage = evidence(text, top)
            row = {
                "split": split,
                "queryid": queryid,
                "model": model,
                "top_docid": top.identity if top else "",
                "has_evidence": bool(passage),
                "overlap": score,
                "top_judged_relevant": bool(top and judged.get(top.identity, 0) > 0),
            }
            for depth in DEPTHS:
                row.update({f"{key}@{depth}": value for key, value in metrics(identities, judged, depth).items()})
            results.append(row)
    return results


def citation_metrics(rows: list[dict], threshold: float) -> dict:
    emitted = [row for row in rows if row["has_evidence"] and row["overlap"] >= threshold]
    matched = sum(row["top_judged_relevant"] for row in emitted)
    precision = matched / len(emitted) if emitted else 0.0
    recall = matched / len(rows) if rows else 0.0
    denom = 0.25 * precision + recall
    return {
        "threshold": threshold,
        "emitted": len(emitted),
        "matched": matched,
        "coverage": len(emitted) / len(rows),
        "judged_relevance_among_emitted": precision if emitted else None,
        "f0.5": 1.25 * precision * recall / denom if denom else 0.0,
    }


def evaluate(database: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with closing(open_index(database)) as db:
        manifest = json.loads(db.execute("SELECT manifest FROM provenance").fetchone()[0])
        train = assess(db, "train")
        fit = {
            model: [row for row in train if row["model"] == model] for model in MODELS
        }
        selected_model = max(MODELS, key=lambda name: mean(row["ndcg@10"] for row in fit[name]))
        threshold = max(
            (citation_metrics(fit[selected_model], candidate) for candidate in THRESHOLDS),
            key=lambda score: (score["f0.5"], score["coverage"], score["threshold"]),
        )["threshold"]
        test = assess(db, "test")
    results = {}
    metric_keys = [f"{name}@{k}" for k in DEPTHS for name in ("ndcg", "recall", "mrr")]
    for split, values in (("train_calibration", train), ("test", test)):
        results[split] = {}
        for model in MODELS:
            rows = [row for row in values if row["model"] == model]
            results[split][model] = {key: round(mean(row[key] for row in rows), 6) for key in metric_keys}
        selected = [row for row in values if row["model"] == selected_model]
        results[split]["citation_gate"] = citation_metrics(selected, threshold)
    summary = {
        "provenance": manifest,
        "selected_model": selected_model,
        "threshold": threshold,
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(train[0]))
        writer.writeheader()
        writer.writerows(train)
        writer.writerows(test)
    return summary
