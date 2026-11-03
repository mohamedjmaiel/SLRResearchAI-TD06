"""Transmission entre les TD (code fourni, à lire).

    TD03 search_results.json        articles trouvés (PaperRecord)
  ∩ TD04 screening_decisions.json   décision FINALE = include
  ∩ document_manifest.json          texte intégral disponible (status = downloaded)
  → corpus du RAG (SourceDocument : pages extraites, provenance conservée)

Le TD05 écrit ensuite artifacts/verified_evidence.json, que consommera le
WritingAgent : un texte rédigé ne pourra citer que des preuves contrôlées.
"""
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from slrresearch.models import PaperRecord
from slrresearch.rag_models import RagRun

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
EVALUATION_PATH = ARTIFACTS / "rag_evaluation.json"
EVIDENCE_PATH = ARTIFACTS / "verified_evidence.json"


# ---------------------------------------------------------------------------
# Contrats des artefacts reçus
# ---------------------------------------------------------------------------

class SearchArtifact(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["TD03"]
    topic: str
    rq: str
    queries: dict[str, str]
    papers: list[PaperRecord] = Field(min_length=1)


class FinalDecision(BaseModel):
    paper_id: str
    final_decision: Literal["include", "exclude"]   # "uncertain" n'est PAS une décision finale
    decider: Literal["regles", "llm", "accord", "humain"]
    reason: str


class ScreeningArtifact(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["TD04"]
    source_artifact: str
    decisions: list[FinalDecision]


class DocumentRecord(BaseModel):
    paper_id: str
    status: Literal["downloaded", "manual_required", "unavailable"]
    source_url: str = ""
    pages_key: str | None = None                     # clé dans extracted_pages.json


class DocumentManifest(BaseModel):
    schema_version: Literal["1.0"]
    origin: Literal["DOCUMENT_PREPARATION"]
    source_artifact: str
    pedagogical_fixture: bool = True
    documents: list[DocumentRecord]


class SourceDocument(BaseModel):
    """Un document du corpus RAG : son identité et ses pages extraites."""

    paper_id: str
    title: str
    pages: list[str] = Field(min_length=1)


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Artefact introuvable : {path}")
    return path.read_text(encoding="utf-8")


def load_rag_handoff(artifact_dir: Path = ARTIFACTS) -> tuple[list[SourceDocument], dict]:
    """Jointure contrôlée par paper_id : trouvés ∩ inclus ∩ disponibles."""
    search = SearchArtifact.model_validate_json(_read(artifact_dir / "search_results.json"))
    screening = ScreeningArtifact.model_validate_json(_read(artifact_dir / "screening_decisions.json"))
    manifest = DocumentManifest.model_validate_json(_read(artifact_dir / "document_manifest.json"))
    extracted = json.loads(_read(artifact_dir / "extracted_pages.json"))

    papers = {paper.id: paper for paper in search.papers}
    unknown = {d.paper_id for d in screening.decisions} - set(papers)
    if unknown:
        raise ValueError(f"Décisions TD04 sur des articles absents du TD03 : {sorted(unknown)}")
    included = {d.paper_id for d in screening.decisions if d.final_decision == "include"}
    available = {d.paper_id: d for d in manifest.documents
                 if d.status == "downloaded" and d.pages_key in extracted}
    selected = sorted(set(papers) & included & set(available))
    documents = [
        SourceDocument(paper_id=pid, title=papers[pid].title, pages=extracted[available[pid].pages_key])
        for pid in selected
    ]
    lineage = {
        "topic": search.topic,
        "rq": search.rq,
        "found": len(papers),
        "included": len(included),
        "documents_available": len(available),
        "rag_documents": len(documents),
        "excluded_from_rag": {
            pid: ("exclu au TD04" if pid not in included else
                  "texte intégral indisponible (" + next(
                      (d.status for d in manifest.documents if d.paper_id == pid), "absent du manifeste") + ")")
            for pid in sorted(set(papers) - set(selected))
        },
        "sources": {
            "search": "TD03/search_results.json",
            "screening": "TD04/screening_decisions.json",
            "documents": "document_manifest.json",
        },
    }
    return documents, lineage


# ---------------------------------------------------------------------------
# Annotations d'évaluation (passages de référence : paper_id + page)
# ---------------------------------------------------------------------------

def _normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.replace("’", "'").split()).rstrip(" ?")


def load_annotations(path: Path = EVALUATION_PATH) -> list[dict]:
    return json.loads(_read(path))["questions"]


def relevant_pages_for(question: str, path: Path = EVALUATION_PATH) -> set[tuple[str, int]] | None:
    """Pages de référence annotées pour cette question ; None si elle n'est pas annotée."""
    wanted = _normalize_question(question)
    for item in load_annotations(path):
        if _normalize_question(item["question"]) == wanted:
            return {(ref["paper_id"], ref["page"]) for ref in item["relevant_pages"]}
    return None


# ---------------------------------------------------------------------------
# Artefact produit pour le TD suivant
# ---------------------------------------------------------------------------

def save_evidence_artifact(run: RagRun, topic: str, retriever: str,
                           path: Path = EVIDENCE_PATH) -> dict:
    """Ajoute (ou remplace) la preuve d'une question dans verified_evidence.json.

    Seules les preuves au statut « verified » sont autorisées pour la rédaction ;
    les autres restent visibles pour l'humain (human_review) ou pour l'audit.
    """
    if run.evidence is None:
        raise ValueError("Aucune preuve à enregistrer (le pipeline s'est abstenu)")
    payload = {"schema_version": "1.0", "origin": "RAG_TD05", "topic": topic,
               "source_artifacts": ["search_results.json", "screening_decisions.json",
                                    "document_manifest.json"],
               "evidence": []}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("topic") == topic:
                payload["evidence"] = previous.get("evidence", [])
        except json.JSONDecodeError:
            pass
    record = {
        "question": run.question,
        "retrieval_query": run.retrieval_query,
        "retriever": retriever,
        "retrieved_chunk_ids": [c["chunk_id"] for c in run.selected_chunks],
        **run.evidence.model_dump(),
        "authorized_for_writing": run.evidence.status == "verified",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload["evidence"] = [e for e in payload["evidence"] if e.get("question") != run.question]
    payload["evidence"].append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
