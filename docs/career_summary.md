# Career summary

**FR**

- Conception d’un moteur BM25 et d’une expérience de fusion de rang sur 5 183 résumés SciFact.
- Évaluation sur 300 requêtes avec nDCG@10 de 0,677 pour BM25 et seuil d’abstention calibré hors test.

**EN**

- Built an FTS5/BM25 scientific retrieval index and a rank-fusion experiment over 5,183 SciFact abstracts.
- Measured 0.677 test nDCG@10 over 300 queries and calibrated a separate citation abstention rule.

## Demonstrated skills

Search indexing, retrieval baselines, split auditing, judged relevance metrics, reproducibility and extractive citations.

## Interview questions

- What ranking behavior changes when an exact-bigram search supplies half of the reciprocal-rank signal?
- How can normalization uncover leakage even when split IDs are disjoint?
- Why are judged relevance and textual entailment different outcomes?
- How much coverage is lost when a local overlap gate abstains?

## Two-minute explanation

The project indexes science abstracts with SQLite FTS5. It compares weighted BM25 against a reciprocal-rank fusion with adjacent word pairs. Model selection and passage-overlap threshold selection use 807 calibration queries. BM25 achieved 0.677 nDCG@10 on the 300 official test queries. The extractive mode returns a document ID and passage only when query overlap passes the calibrated threshold.

## Five-minute explanation

Start with the archive hash and qrels checks. Show how title terms and abstracts affect weighted BM25, then describe how RRF redistributes rank based on exact phrases. Walk through the two overlapping claim texts removed from calibration and the weighted F-score chosen for the threshold. The gate returns evidence for 163 queries, with 119 judged relevant top documents. Explain that missing judgments count as unobserved relevance and that the benchmark scores articles, while the passage itself remains a candidate. Compare dense retrieval and passage-level judging as distinct follow-up experiments.

## Follow-up experiments

Test a compact dense ranker and exact sentence judgments using the upstream evidence annotations. Add a query-level review set with adjudication of new evidence candidates.
