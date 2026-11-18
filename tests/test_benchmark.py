"""Local synthetic fixtures test contracts without downloading a corpus."""
from contextlib import closing
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from corpus import build, digest, open_index
from evaluation import citation_metrics, evaluate, metrics
from retrieval import Hit, evidence, search


def fixture(path: Path, invalid_doc: bool = False) -> None:
    documents = [
        {"_id": "a", "title": "Lake nutrients", "text": "Nitrogen loading increased in lake water."},
        {"_id": "b", "title": "Mountain soil carbon", "text": "Soil carbon declined on steep slopes."},
    ]
    queries = [
        {"_id": "1", "text": "lake nutrients increased"},
        {"_id": "2", "text": "soil carbon declined"},
        {"_id": "3", "text": "Soil   carbon declined"},
    ]
    with ZipFile(path, "w") as archive:
        for filename, records in (("corpus.jsonl", documents), ("queries.jsonl", queries)):
            archive.writestr(f"scifact/{filename}", "\n".join(json.dumps(record) for record in records))
        header = "query-id\tcorpus-id\tscore\n"
        archive.writestr("scifact/qrels/train.tsv", header + "1\ta\t1\n3\tb\t1\n")
        key = "missing" if invalid_doc else "b"
        archive.writestr("scifact/qrels/test.tsv", header + f"2\t{key}\t1\n")


class RetrievalTests(unittest.TestCase):
    def test_end_to_end_index_metrics_and_overlap_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "corpus.zip"
            index = root / "index.sqlite"
            fixture(path)
            manifest = build(path, index, digest(path))
            self.assertEqual(manifest["train_text_overlaps_removed"], 1)
            self.assertEqual(manifest["train_queries"], 1)
            with closing(open_index(index)) as db:
                self.assertEqual(search(db, "soil carbon declined")["bm25"][0].identity, "b")
                self.assertEqual(search(db, '""" OR * : the _')["bm25"], [])
            result = evaluate(index, root / "results")
            self.assertEqual(result["results"]["test"]["bm25"]["recall@10"], 1.0)
            self.assertEqual(result["provenance"]["test_queries"], 1)

    def test_invalid_qrel_cannot_replace_an_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.zip"
            index = Path(folder) / "index.sqlite"
            index.write_bytes(b"previous index")
            fixture(path, invalid_doc=True)
            with self.assertRaisesRegex(ValueError, "Invalid qrel"):
                build(path, index, digest(path))
            self.assertEqual(index.read_bytes(), b"previous index")

    def test_rank_metrics(self):
        scored = metrics(["z", "a", "b"], {"a": 1, "b": 1}, 2)
        self.assertEqual(scored["recall"], 0.5)
        self.assertEqual(scored["mrr"], 0.5)
        self.assertAlmostEqual(scored["ndcg"], 0.3868528072)

    def test_citation_evidence_and_abstention(self):
        hit = Hit("a", "Lake sediments", "Calcium was present. Lake nitrate loads changed.")
        overlap, passage = evidence("nitrate loads changed", hit)
        self.assertEqual((overlap, passage), (1.0, "Lake nitrate loads changed."))
        self.assertEqual(evidence("a or the", hit), (0.0, ""))
        gated = citation_metrics([
            {"overlap": 0.75, "has_evidence": True, "top_judged_relevant": True},
            {"overlap": 0.25, "has_evidence": True, "top_judged_relevant": False},
        ], 0.5)
        self.assertEqual(gated["emitted"], 1)
        self.assertEqual(gated["coverage"], 0.5)
        self.assertEqual(gated["judged_relevance_among_emitted"], 1.0)
