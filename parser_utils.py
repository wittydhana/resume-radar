from pathlib import Path
from docx import Document
from pypdf import PdfReader

def extract_text(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if s == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    if s == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""