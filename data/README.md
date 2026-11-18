# Data

This project uses BEIR's SciFact packaging of the Allen Institute for AI benchmark. It downloads the original ZIP from the BEIR mirror, checks its SHA-256 and reads JSONL and TSV members directly. The tested archive has 2,816,079 compressed bytes, 5,183 abstracts, 1,109 query records, 809 official calibration queries and 300 official test queries. The source archive SHA-256 is `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`. [BEIR dataset card](https://huggingface.co/datasets/BeIR/scifact) · [SciFact](https://github.com/allenai/scifact)

Fields include corpus ID, title and abstract, query ID and claim text, and qrels linking queries to graded document relevance. The build checks archive members, schema, missing references and IDs. It preserves the official test set. Two normalized claim texts overlapped with calibration and were removed from the calibration side before model and threshold selection.

The originators license SciFact claims and annotations under CC BY 4.0 and the S2ORC corpus under ODC-BY 1.0. The original code is Apache-2.0. This repository's code is MIT. Keep raw data and indexes local. [Upstream terms by component](https://github.com/allenai/scifact/blob/master/LICENSE.md)
