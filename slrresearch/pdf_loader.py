"""Extraction page par page d'un PDF (code fourni, mode projet).

Dans le TD, les pages des articles retenus sont déjà extraites dans
artifacts/extracted_pages.json. Dans votre projet, la même liste de pages
s'obtient avec extract_pdf_pages("documents/W1001.pdf"), puis alimente
chunk_pages() sans autre changement. La page reste une métadonnée de
l'extraction : elle n'est jamais devinée.
"""
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


def extract_pdf_pages(source: str | Path | BinaryIO) -> list[str]:
    """Texte de chaque page (index 0 = page 1) ; signale un PDF sans couche texte."""
    reader = PdfReader(source)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not pages:
        raise ValueError("Le PDF ne contient aucune page")
    if not any(pages):
        raise ValueError("Aucun texte extractible : PDF scanné, un OCR est nécessaire")
    return pages
