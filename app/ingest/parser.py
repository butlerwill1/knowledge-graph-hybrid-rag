"""Parse text files and PDF text layers into page-level document records."""

from __future__ import annotations

from pathlib import Path

from app.models.schemas import DocumentRecord, ParsedDocument, ParsedPage


class PDFParser:
    """Parse supported source files into page-level text records."""

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF text layer or UTF-8 text file into a new document record."""

        suffix = file_path.suffix.lower()
        document = DocumentRecord(title=file_path.stem, source_path=str(file_path))

        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            return ParsedDocument(document=document, pages=[ParsedPage(page_number=1, text=text)])

        if suffix != ".pdf":
            raise ValueError(f"Unsupported file type: {suffix}")

        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF parsing. Install the 'pdf' extra.") from exc

        # PyMuPDF preserves page boundaries, which keeps downstream citations tied
        # to the source page even when the PDF layout itself is imperfect.
        doc = fitz.open(file_path)
        pages = [
            ParsedPage(page_number=index + 1, text=page.get_text("text"))
            for index, page in enumerate(doc)
        ]
        return ParsedDocument(document=document, pages=pages)
