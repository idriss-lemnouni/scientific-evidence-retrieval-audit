"""Lexical retrieval, rank fusion and extractive citation candidates."""
from dataclasses import dataclass
import re
import sqlite3
import unicodedata

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have how in into is it its may might of on or our should so than that the their
them then there these they this those to was were what when where which who
will with would you your
""".split())
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MODELS = ("bm25", "phrase-fusion")
DEPTH = 50


@dataclass(frozen=True)
class Hit:
    identity: str
    title: str
    body: str


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return TOKEN.findall("".join(char for char in normalized if not unicodedata.combining(char)))


def fts_rank(db: sqlite3.Connection, expressions: list[str]) -> list[Hit]:
    if not expressions:
        return []
    query = " OR ".join(f'"{expression}"' for expression in expressions)
    return [
        Hit(*row) for row in db.execute("""
            SELECT docid, title, body FROM documents
            WHERE documents MATCH ?
            ORDER BY bm25(documents, 0, 2.5, 1), docid LIMIT ?
        """, (query, DEPTH))
    ]


def search(db: sqlite3.Connection, query: str) -> dict[str, list[Hit]]:
    tokens = tokenize(query)[:60]
    terms = list(dict.fromkeys(token for token in tokens if token not in STOPWORDS))
    lexical = fts_rank(db, terms)
    phrases = list(dict.fromkeys(
        f"{left} {right}" for left, right in zip(tokens, tokens[1:])
        if left not in STOPWORDS and right not in STOPWORDS
    ))
    if not lexical or not phrases:
        return {"bm25": lexical, "phrase-fusion": lexical}
    phrase_hits = fts_rank(db, phrases)
    scores = {}
    documents = {}
    for hits in (lexical, phrase_hits):
        for rank, hit in enumerate(hits, 1):
            scores[hit.identity] = scores.get(hit.identity, 0.0) + 1 / (60 + rank)
            documents[hit.identity] = hit
    return {
        "bm25": lexical,
        "phrase-fusion": [documents[key] for key in sorted(scores, key=lambda key: (-scores[key], key))],
    }


def evidence(query: str, hit: Hit | None) -> tuple[float, str]:
    if hit is None:
        return 0.0, ""
    terms = set(tokenize(query)).difference(STOPWORDS)
    if not terms:
        return 0.0, ""
    passages = [hit.title, *SENTENCE.split(hit.body)]
    sentence = max(passages, key=lambda text: len(terms.intersection(tokenize(text))))
    coverage = len(terms.intersection(tokenize(sentence))) / len(terms)
    return coverage, sentence.strip()
