"""Assemble deterministic, evidence-grounded query responses."""

from __future__ import annotations

from app.models.schemas import EvidenceSnippet, QueryResponse


class GroundedAnswerService:
    """Build a transparent draft response directly from retrieved evidence."""

    def build_response(self, question: str, evidence: list[EvidenceSnippet], graph_facts: list[str]) -> QueryResponse:
        """Return a deterministic answer plus all evidence and graph facts."""

        if not evidence:
            answer = "Insufficient evidence was retrieved to answer the question."
            return QueryResponse(answer=answer, evidence=[], graph_facts=graph_facts)

        # Keep the draft short while returning the complete selected evidence
        # separately in the response model.
        top_sources = evidence[:3]
        citations = ", ".join(f"page {item.page}" for item in top_sources)
        summary = " ".join(item.text[:240].strip() for item in top_sources)
        answer = (
            f"Grounded draft answer for: '{question}'. "
            f"The strongest retrieved evidence comes from {citations}. "
            f"Relevant source text: {summary}"
        )
        return QueryResponse(answer=answer, evidence=evidence, graph_facts=graph_facts)
