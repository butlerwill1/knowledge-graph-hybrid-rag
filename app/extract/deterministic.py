from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.models.schemas import ChunkRecord, ClaimRecord, EntityRecord, GraphEdge

if TYPE_CHECKING:
    import spacy as spacy_type

# spaCy entity label → our canonical label
_SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organisation",
    "GPE": "Location",
    "LOC": "Location",
    "PRODUCT": "Product",
    "WORK_OF_ART": "Work",
    "EVENT": "Event",
    "LAW": "Law",
    "LANGUAGE": "Language",
    "NORP": "Group",
    "FAC": "Facility",
}

# Verbs that are semantically meaningful as relation types in research text.
# Anything not in this set falls back to CO_OCCURS_WITH.
_ALLOWED_RELATIONS = frozenset({
    "achieve", "adopt", "annotate", "apply", "augment", "base", "bootstrap",
    "build", "combine", "compare", "compress", "conduct", "demonstrate",
    "design", "develop", "eliminate", "embody", "enable", "establish",
    "evaluate", "examine", "exploit", "explore", "extend", "finetune",
    "follow", "generalize", "generate", "guide", "hallucinate", "hypothesize",
    "identify", "improve", "incorporate", "indicate", "integrate", "introduce",
    "investigate", "involve", "lack", "learn", "leverage", "limit",
    "outperform", "overcome", "perform", "pioneer", "plan", "present",
    "process", "produce", "propose", "provide", "reduce", "represent",
    "require", "retrieve", "show", "simulate", "solve", "study", "summarize",
    "support", "surpass", "test", "train", "transform", "tune", "understand",
    "use", "utilize",
})

# Minimum character length for a claim sentence
_MIN_CLAIM_LEN = 40

# Verbs that signal a factual assertion (used for claim extraction)
_CLAIM_VERBS = frozenset({"is", "are", "was", "were", "has", "have", "can", "should", "must", "show", "demonstrate",
                           "propose", "achieve", "improve", "outperform", "enable", "reduce", "increase"})


@lru_cache(maxsize=1)
def _load_nlp() -> "spacy_type.Language":
    import spacy
    # en_core_web_sm: ~12 MB vs en_core_web_lg: ~560 MB
    # Word vectors aren't used for NER so the large model wastes RAM
    return spacy.load("en_core_web_sm", disable=["textcat"])


class DeterministicExtractor:
    def extract(self, chunks: list[ChunkRecord]) -> tuple[list[EntityRecord], list[ClaimRecord], list[GraphEdge]]:
        nlp = _load_nlp()

        all_entities: list[EntityRecord] = []
        all_claims: list[ClaimRecord] = []
        all_edges: list[GraphEdge] = []

        texts = [c.text for c in chunks]
        for chunk, doc in zip(chunks, nlp.pipe(texts, batch_size=16)):
            entities, claims, edges = self._process_chunk(chunk, doc)
            all_entities.extend(entities)
            all_claims.extend(claims)
            all_edges.extend(edges)

        return all_entities, all_claims, all_edges

    def _process_chunk(self, chunk: ChunkRecord, doc: object) -> tuple[list[EntityRecord], list[ClaimRecord], list[GraphEdge]]:
        entities: list[EntityRecord] = []
        claims: list[ClaimRecord] = []
        edges: list[GraphEdge] = []
        entity_by_text: dict[str, EntityRecord] = {}

        # --- Entity extraction via NER ---
        for ent in doc.ents:
            label = _SPACY_LABEL_MAP.get(ent.label_, None)
            if label is None:
                continue
            key = ent.text.strip().lower()
            if not key or len(key) < 3:
                continue
            if key in entity_by_text:
                continue

            record = EntityRecord(
                canonical_name=ent.text.strip(),
                label=label,
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                page=chunk.page,
                confidence=0.75,
                extractor="spacy",
                source_text=ent.sent.text.strip(),
            )
            entity_by_text[key] = record
            entities.append(record)
            edges.append(GraphEdge(
                source_id=chunk.id,
                target_id=record.id,
                relation="MENTIONS",
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                page=chunk.page,
                confidence=0.75,
                extractor="spacy",
            ))

        # --- Claim extraction via sentence assertions ---
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text) < _MIN_CLAIM_LEN:
                continue
            lowered = text.lower()
            if not any(f" {v} " in f" {lowered} " for v in _CLAIM_VERBS):
                continue

            claim = ClaimRecord(
                text=text,
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                page=chunk.page,
                confidence=0.60,
                extractor="spacy",
                source_text=text,
            )
            claims.append(claim)
            edges.append(GraphEdge(
                source_id=chunk.id,
                target_id=claim.id,
                relation="MAKES_CLAIM",
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                page=chunk.page,
                confidence=0.60,
                extractor="spacy",
            ))

            # Link claim to entities mentioned in the same sentence
            for ent in sent.ents:
                key = ent.text.strip().lower()
                if key in entity_by_text:
                    edges.append(GraphEdge(
                        source_id=claim.id,
                        target_id=entity_by_text[key].id,
                        relation="ABOUT",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        confidence=0.60,
                        extractor="spacy",
                    ))

        # --- Entity-to-entity relations via dependency parsing ---
        for sent in doc.sents:
            sent_ents = [e for e in sent.ents if _SPACY_LABEL_MAP.get(e.label_) and e.text.strip().lower() in entity_by_text]
            if len(sent_ents) < 2:
                continue
            for i, src_ent in enumerate(sent_ents):
                for tgt_ent in sent_ents[i + 1:]:
                    relation = _extract_relation(src_ent, tgt_ent)
                    src_record = entity_by_text[src_ent.text.strip().lower()]
                    tgt_record = entity_by_text[tgt_ent.text.strip().lower()]
                    edges.append(GraphEdge(
                        source_id=src_record.id,
                        target_id=tgt_record.id,
                        relation=relation,
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        confidence=0.55,
                        extractor="spacy",
                    ))

        return entities, claims, edges


def _extract_relation(src_ent: object, tgt_ent: object) -> str:
    """Find the most meaningful verb on the dependency path between two entities."""
    src_root = src_ent.root
    tgt_root = tgt_ent.root

    verbs: list[str] = []
    for root in (src_root, tgt_root):
        token = root
        for _ in range(6):
            if token.pos_ == "VERB" and token.lemma_ in _ALLOWED_RELATIONS:
                verbs.append(token.lemma_.upper())
            if token.head == token:
                break
            token = token.head
        if verbs:
            break

    return verbs[0] if verbs else "CO_OCCURS_WITH"
