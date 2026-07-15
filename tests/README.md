# Tests

The test suite currently consists of focused smoke tests in `test_smoke.py`.

Run it from the repository root:

```powershell
python -m pytest -q
```

## Current Coverage

- File-vector retrieval and deterministic answer assembly.
- LLM extraction remaining disabled when no API key is present.
- Translation of structured LLM entities, claims, relations, and source works into graph records.
- OpenRouter-compatible structured chat request construction.
- Token, cost, and validation-count logging.
- Adjacent chunk context being supplied to the prompt.
- Resolved `canonical_name` and original `mention_text` behaviour.
- Rejection of vague entity names and dependent claims.

The OpenRouter request test uses a fake completion client and does not spend API credit.

## Important Gaps

- No integration test currently starts Neo4j.
- No API route tests exercise multipart uploads or query requests.
- No parser tests cover malformed, scanned, encrypted, or layout-heavy PDFs.
- No end-to-end ingestion test verifies all generated files and database writes together.
- No retry or provider-failure tests exist.
- No reviewed extraction fixture measures entity, claim, or relation precision.
