"""Build, query and evaluate a citation-aware SciFact retrieval benchmark."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from zipfile import BadZipFile

from corpus import SOURCE_SHA256, build, download, open_index
from evaluation import evaluate
from retrieval import MODELS, evidence, search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("fetch", "build"):
        command = commands.add_parser(name)
        command.add_argument("--archive", required=True, type=Path)
        command.add_argument("--sha256", default=SOURCE_SHA256)
    commands.choices["build"].add_argument("--index", required=True, type=Path)
    for name in ("evaluate", "search", "answer"):
        commands.add_parser(name).add_argument("--index", required=True, type=Path)
    commands.choices["evaluate"].add_argument("--out", required=True, type=Path)
    commands.choices["search"].add_argument("query")
    commands.choices["search"].add_argument("--model", choices=MODELS, default="bm25")
    commands.choices["answer"].add_argument("query")
    commands.choices["answer"].add_argument("--calibration", required=True, type=Path)
    options = parser.parse_args()
    try:
        if options.action == "fetch":
            download(options.archive, options.sha256)
            result = {"archive": str(options.archive), "sha256": options.sha256}
        elif options.action == "build":
            result = build(options.archive, options.index, options.sha256)
        elif options.action == "evaluate":
            result = evaluate(options.index, options.out)
        else:
            with closing(open_index(options.index)) as db:
                rankings = search(db, options.query)
                if options.action == "search":
                    result = [
                        {"corpus_id": hit.identity, "title": hit.title}
                        for hit in rankings[options.model][:10]
                    ]
                else:
                    config = json.loads(options.calibration.read_text(encoding="utf-8"))
                    manifest = json.loads(db.execute("SELECT manifest FROM provenance").fetchone()[0])
                    if config["provenance"]["archive_sha256"] != manifest["archive_sha256"]:
                        raise ValueError("Calibration archive does not match the index")
                    threshold = float(config["threshold"])
                    if not 0 < threshold <= 1:
                        raise ValueError("Calibration threshold must be in (0, 1]")
                    hits = rankings[config["selected_model"]]
                    hit = hits[0] if hits else None
                    score, passage = evidence(options.query, hit)
                    result = {"status": "abstained", "lexical_overlap": round(score, 4)}
                    if hit and passage and score >= threshold:
                        result.update(
                            status="citation_candidate",
                            corpus_id=hit.identity,
                            title=hit.title,
                            passage=" ".join(passage.split()[:25]),
                            truncated=len(passage.split()) > 25,
                            interpretation="Retrieved passage. Verify relevance against the source.",
                        )
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, sqlite3.Error, BadZipFile) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
