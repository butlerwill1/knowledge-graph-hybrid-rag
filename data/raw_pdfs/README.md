# Raw Source Files

The upload endpoint saves incoming files here before ingestion. The name is historical: both `.pdf` and `.txt` sources are supported by the parser.

PDF and JSON-like runtime data are ignored by Git, so source documents remain local by default.

The current API writes the uploaded filename directly into this directory. Do not treat the endpoint as a hardened public upload service without adding filename sanitisation, file-size limits, content validation, malware controls, and access controls.

For image-only or scanned PDFs, the current parser will not perform OCR. Preprocess those files with OCR or add a vision/OCR parsing path before expecting useful extraction.
