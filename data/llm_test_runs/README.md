# LLM Test Runs

This directory contains manually launched extraction experiments used to inspect prompts, models, costs, validation, and graph quality before importing anything into Neo4j.

These files are not part of the normal ingestion pipeline and are not imported automatically.

Test-run structures can vary, but recent checkpointed runs contain:

- Run status and timestamps.
- Requested, attempted, successful, and failed chunk counts.
- Per-chunk status and errors.
- Model and token usage totals.
- Actual OpenRouter cost totals when available.
- Validation rejection counts.
- Extracted entity, claim, and edge arrays.

The checkpointed runner writes the file after every attempted chunk. A run can therefore have `status="complete_with_errors"` and still contain useful partial results.

Before importing a run, check:

1. Every expected chunk succeeded or was deliberately excluded.
2. Vague and context-dependent entities have been reviewed.
3. `ABOUT` and `FROM_WORK` relationships have the intended meaning.
4. Duplicate canonical names have an explicit canonicalisation strategy.
5. The target Neo4j database does not already contain the same generated IDs or equivalent records.

JSON files in this directory are ignored by Git.
