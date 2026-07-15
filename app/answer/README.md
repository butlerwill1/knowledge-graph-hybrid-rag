# Answer Assembly

This package turns retrieved evidence into the API's `QueryResponse`.

`GroundedAnswerService` currently provides a deterministic draft rather than invoking an LLM. It:

- Returns an insufficient-evidence response when retrieval found nothing.
- Uses the top three evidence records to create page references.
- Concatenates shortened source snippets into a draft response.
- Returns all selected evidence and graph facts alongside the answer.

This keeps the prototype grounded and testable while retrieval is under development, but it is not yet natural-language answer synthesis. A future generator should preserve the evidence objects, cite them explicitly, and refuse unsupported conclusions.
