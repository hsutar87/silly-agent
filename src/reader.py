import logging
from pathlib import Path
from typing import Dict, Any
import fitz  # PyMuPDF
from docx import Document as DocxLoader

class UniversalReader:
    """Reads PDF, DOCX, and TXT files and returns text with metadata."""
    
    def read_file(self, path: Path) -> Dict[str, Any]:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                return self._read_pdf(path)
            elif ext == ".docx":
                return self._read_docx(path)
            elif ext in [".txt", ".md"]:
                return self._read_text(path)
            else:
                raise ValueError(f"Unsupported format: {ext}")
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
            raise

    def _read_pdf(self, path: Path) -> Dict[str, Any]:
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return {"content": text, "metadata": {"source": path.name, "type": "pdf"}}

    def _read_docx(self, path: Path) -> Dict[str, Any]:
        doc = DocxLoader(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return {"content": text, "metadata": {"source": path.name, "type": "docx"}}

    def _read_text(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read(), "metadata": {"source": path.name, "type": "text"}}