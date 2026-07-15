# Ingestion

This package turns a local source file into parsed pages, chunks, extracted graph records, vector records, and audit JSON.

## Components

### `parser.py`

`PDFParser` creates a new `DocumentRecord` and parses either:

- `.pdf` through PyMuPDF using `page.get_text("text")`.
- `.txt` as one UTF-8 page.

The parser uses only a PDF's embedded text layer. It does not run OCR or a vision model, so scanned pages can produce empty or poor text.

### `chunker.py`

`SectionAwareChunker` splits each page on blank-line paragraph boundaries and combines paragraphs up to a nominal target of 1,200 characters.

Chunks do not span pages. A single paragraph longer than the target remains long because the current implementation does not split inside a paragraph. The `section` field currently remains at its default `Body`; heading detection is not implemented.

### `pipeline.py`

`IngestionPipeline` coordinates the complete write path:

1. Parse the source file.
2. Chunk the parsed pages.
3. Run deterministic extraction.
4. Run optional LLM extraction.
5. Combine both sets of graph records.
6. Upsert the document, chunks, entities, claims, and edges into the configured graph store.
7. Add chunks to the local vector store.
8. Write parsed and extracted JSON audit files.
9. Return an `IngestionResult` containing created-record counts.

The graph writes and file writes are not wrapped in one transaction. A failure partway through ingestion can therefore leave partial state and should be retried carefully.

## Outputs

| Output | Destination |
| --- | --- |
| Uploaded source | `RAW_DATA_DIR` |
| Parsed document | `PARSED_DATA_DIR/<document-id>.json` |
| Combined extraction | `EXTRACTED_DATA_DIR/<document-id>.json` |
| Chunk vectors | `VECTOR_STORE_PATH` |
| Graph records | Configured `GraphStore` |
| LLM usage | `LLM_USAGE_LOG_PATH` |
