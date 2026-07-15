# Evaluation

This package contains early retrieval and answer-quality metrics.

`metrics.py` currently implements `unsupported_answer_rate`, defined as the proportion of `QueryResponse` objects containing no evidence. An empty evaluation set returns `0.0`.

This is not yet a complete evaluation framework. Useful future metrics include:

- Entity and relation precision against a reviewed extraction set.
- Claim grounding and evidence-page accuracy.
- Canonicalisation precision and duplicate-entity rate.
- Retrieval recall at different `top_k` values.
- Answer faithfulness and citation coverage.
- Extraction cost and latency by model and document type.
