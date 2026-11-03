"""Contrats de données du TD05 - RAG classique (code fourni, à lire).

Chunk            : l'unité de récupération ; elle garde sa provenance.
ProposedEvidence : ce que le LLM PROPOSE (contrat de sortie structurée).
VerifiedEvidence : la proposition + le statut attribué PAR LE PROGRAMME.
RagRun           : la trace complète d'une exécution du pipeline.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

EvidenceStatus = Literal["verified", "human_review", "rejected"]


def paper_slug(paper_id: str) -> str:
    """Identifiant court et stable d'un article, utilisé dans les chunk_id.

    "https://openalex.org/W1001" -> "W1001" ; "s2-7f3a" -> "s2-7f3a"
    """
    last = paper_id.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9-]+", "-", last).strip("-") or "doc"


class Chunk(BaseModel):
    """Un passage indexé : identité + localisation + texte."""

    paper_id: str = Field(min_length=1)       # identifiant transmis depuis le TD03
    chunk_id: str = Field(min_length=1)       # ex. W1001_p3_c2
    page: int = Field(ge=1)                   # page RÉELLE issue de l'extraction
    text: str = Field(min_length=1)


class ProposedEvidence(BaseModel):
    """Sortie structurée demandée au LLM : une réponse et UNE citation localisée."""

    answer: str = Field(description="Réponse courte à la question, en français")
    supporting_quote: str = Field(description="Extrait recopié mot pour mot du passage cité")
    paper_id: str = Field(description="Valeur PAPER du passage cité")
    chunk_id: str = Field(description="Valeur CHUNK du passage cité")
    page: int = Field(ge=1, description="Valeur PAGE du passage cité")


class VerifiedEvidence(ProposedEvidence):
    """Proposition du LLM + décision du programme (jamais du LLM)."""

    status: EvidenceStatus
    verification_reason: str


class RagRun(BaseModel):
    """Trace d'une exécution : ce qui a été cherché, transmis, proposé, contrôlé."""

    question: str                                  # RQ posée au générateur
    retrieval_query: str                           # requête envoyée au retriever
    selected_chunks: list[dict] = Field(default_factory=list)
    context: str = ""
    evidence: VerifiedEvidence | None = None
    abstained: bool = False
    reason: str = ""
