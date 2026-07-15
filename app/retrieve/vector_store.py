"""Store chunk vectors in JSON and rank them with cosine similarity."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from app.models.schemas import ChunkRecord


class FileVectorStore:
    """Persist simple hashed chunk vectors in a local JSON file."""

    def __init__(self, store_path: Path, dimensions: int = 256) -> None:
        """Load an existing store or initialise an empty in-memory record map."""

        self.store_path = store_path
        self.dimensions = dimensions
        self.records: dict[str, dict[str, object]] = {}
        self._load()

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Embed and upsert chunks by ID, then persist the complete store."""

        for chunk in chunks:
            self.records[chunk.id] = {
                "chunk": chunk.model_dump(mode="json"),
                "embedding": self._embed(chunk.text),
            }
        self._persist()

    def search(self, query: str, limit: int = 5) -> list[tuple[ChunkRecord, float]]:
        """Rank stored chunks by cosine similarity to the query vector."""

        query_embedding = self._embed(query)
        scored: list[tuple[ChunkRecord, float]] = []
        for record in self.records.values():
            score = self._cosine_similarity(query_embedding, record["embedding"])
            chunk = ChunkRecord.model_validate(record["chunk"])
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _embed(self, text: str) -> list[float]:
        """Create a normalised hashed bag-of-words vector."""

        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            # This baseline deliberately avoids an embedding dependency. Python's
            # process-randomised hash makes it unsuitable for durable production vectors.
            index = hash(token) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _cosine_similarity(self, left: list[float], right: object) -> float:
        """Return a dot product for normalised vectors, or zero for invalid data."""

        if not isinstance(right, list):
            return 0.0
        return sum(left_value * float(right_value) for left_value, right_value in zip(left, right, strict=False))

    def _load(self) -> None:
        """Load records from disk when the configured store already exists."""

        if not self.store_path.exists():
            return
        self.records = json.loads(self.store_path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        """Rewrite the JSON store with the current in-memory records."""

        self.store_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
