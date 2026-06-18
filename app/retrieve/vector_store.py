from __future__ import annotations

import json
import math
import re
from pathlib import Path

from app.models.schemas import ChunkRecord


class FileVectorStore:
    def __init__(self, store_path: Path, dimensions: int = 256) -> None:
        self.store_path = store_path
        self.dimensions = dimensions
        self.records: dict[str, dict[str, object]] = {}
        self._load()

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        for chunk in chunks:
            self.records[chunk.id] = {
                "chunk": chunk.model_dump(mode="json"),
                "embedding": self._embed(chunk.text),
            }
        self._persist()

    def search(self, query: str, limit: int = 5) -> list[tuple[ChunkRecord, float]]:
        query_embedding = self._embed(query)
        scored: list[tuple[ChunkRecord, float]] = []
        for record in self.records.values():
            score = self._cosine_similarity(query_embedding, record["embedding"])
            chunk = ChunkRecord.model_validate(record["chunk"])
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            index = hash(token) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _cosine_similarity(self, left: list[float], right: object) -> float:
        if not isinstance(right, list):
            return 0.0
        return sum(left_value * float(right_value) for left_value, right_value in zip(left, right, strict=False))

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        self.records = json.loads(self.store_path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        self.store_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

