"""Load the official BEIR SciFact splits and build a local FTS5 index."""
from contextlib import closing
import csv
import hashlib
from io import TextIOWrapper
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

SOURCE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SOURCE_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
MEMBERS = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(path: Path, expected: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as folder:
        target = Path(folder) / path.name
        with urlopen(SOURCE_URL, timeout=45) as response, target.open("wb") as stream:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 20_000_000:
                    raise ValueError("Compressed source exceeded 20 MB")
                stream.write(chunk)
        check_archive(target, expected)
        target.replace(path)


def check_archive(path: Path, expected: str | None) -> str:
    checksum = digest(path)
    if expected and checksum != expected:
        raise ValueError(f"Source SHA-256 mismatch: {checksum}")
    with ZipFile(path) as archive:
        for member in MEMBERS:
            item = archive.getinfo(f"scifact/{member}")
            if item.file_size > 20_000_000:
                raise ValueError(f"Uncompressed {member} exceeded 20 MB")
    return checksum


def read_jsonl(archive: ZipFile, member: str, fields: tuple[str, ...]) -> dict:
    output = {}
    with archive.open(f"scifact/{member}") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            identity = record.get("_id")
            values = tuple(record.get(field) for field in fields)
            if (not isinstance(identity, str) or not identity or identity in output
                    or any(not isinstance(value, str) for value in values)
                    or not any(value.strip() for value in values)):
                raise ValueError(f"Invalid or duplicate identifier/content at {member}:{line_number}")
            output[identity] = values
    if not output:
        raise ValueError(f"{member} is empty")
    return output


def read_qrels(archive: ZipFile, split: str, query_ids: set, corpus_ids: set) -> dict:
    references = {}
    with archive.open(f"scifact/qrels/{split}.tsv") as member:
        reader = csv.DictReader(TextIOWrapper(member, encoding="utf-8"), delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Malformed qrels header in {split}")
        for row in reader:
            query, document, relevance = row["query-id"], row["corpus-id"], int(row["score"])
            if query not in query_ids or document not in corpus_ids or not 0 <= relevance <= 10:
                raise ValueError(f"Invalid qrel: {split} {query} {document}")
            judged = references.setdefault(query, {})
            if document in judged:
                raise ValueError(f"Duplicate qrel: {split} {query} {document}")
            judged[document] = relevance
    if not references or any(not any(judged.values()) for judged in references.values()):
        raise ValueError(f"At least one positive judgement is required per query in {split}")
    return references


def normalized_claim(text: str) -> str:
    return " ".join(text.casefold().split())


def build(path: Path, database: Path, expected: str | None = SOURCE_SHA256) -> dict:
    checksum = check_archive(path, expected)
    with ZipFile(path) as archive:
        documents = read_jsonl(archive, "corpus.jsonl", ("title", "text"))
        queries = read_jsonl(archive, "queries.jsonl", ("text",))
        qrels = {
            split: read_qrels(archive, split, set(queries), set(documents))
            for split in ("train", "test")
        }
    if qrels["train"].keys() & qrels["test"].keys():
        raise ValueError("Train and test share query identifiers")
    test_texts = {normalized_claim(queries[key][0]) for key in qrels["test"]}
    original_train_count = len(qrels["train"])
    qrels["train"] = {
        key: judged for key, judged in qrels["train"].items()
        if normalized_claim(queries[key][0]) not in test_texts
    }
    if not qrels["train"]:
        raise ValueError("No calibration queries remain after removing test-text overlaps")
    manifest = {
        "archive_sha256": checksum,
        "archive_bytes": path.stat().st_size,
        "documents": len(documents),
        "query_records": len(queries),
        "train_queries": len(qrels["train"]),
        "train_text_overlaps_removed": original_train_count - len(qrels["train"]),
        "test_queries": len(qrels["test"]),
    }
    database.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=database.parent) as folder:
        staged = Path(folder) / database.name
        with closing(sqlite3.connect(staged)) as db, db:
            db.executescript("""
                CREATE VIRTUAL TABLE documents USING fts5(
                    docid UNINDEXED, title, body, tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TABLE queries (queryid TEXT PRIMARY KEY, text TEXT NOT NULL);
                CREATE TABLE qrels (
                    split TEXT, queryid TEXT, docid TEXT, score INTEGER,
                    PRIMARY KEY (split, queryid, docid)
                );
                CREATE TABLE provenance (manifest TEXT NOT NULL);
            """)
            db.executemany("INSERT INTO documents VALUES (?, ?, ?)", (
                (identity, *values) for identity, values in documents.items()
            ))
            db.executemany("INSERT INTO queries VALUES (?, ?)", (
                (identity, *values) for identity, values in queries.items()
            ))
            db.executemany("INSERT INTO qrels VALUES (?, ?, ?, ?)", (
                (split, query, doc, grade)
                for split, by_query in qrels.items()
                for query, refs in by_query.items()
                for doc, grade in refs.items()
            ))
            db.execute("INSERT INTO provenance VALUES (?)", (json.dumps(manifest),))
        staged.replace(database)
    return manifest


def open_index(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
