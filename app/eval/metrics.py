from __future__ import annotations

from app.models.schemas import QueryResponse


def unsupported_answer_rate(responses: list[QueryResponse]) -> float:
    if not responses:
        return 0.0
    unsupported = sum(1 for response in responses if not response.evidence)
    return unsupported / len(responses)

